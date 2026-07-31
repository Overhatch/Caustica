package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.gen.BloomPushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * Half-resolution scene-referred bloom prefilter and one separable Gaussian blur.
 * Bloom stays outside the temporal reconstruction and exposure histogram; only the display mapper
 * consumes the completed low-frequency image.
 */
public final class RtBloomPipeline {
    private static final String SHADER = "/caustica/rt/bloom.comp.spv";

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long descriptorSet;
    private final long pipelineLayout;
    private final long pipeline;
    private final long sampler;
    private long boundRtView;
    private long boundExposureView;
    private long boundBloomAView;
    private long boundBloomBView;
    private boolean destroyed;

    private RtBloomPipeline(RtContext ctx, long descriptorSetLayout, long descriptorPool,
                            long descriptorSet, long pipelineLayout, long pipeline, long sampler) {
        this.ctx = ctx;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSet = descriptorSet;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
        this.sampler = sampler;
    }

    public static RtBloomPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(6, stack);
            for (int i = 0; i < 4; i++) {
                bindings.get(i).binding(i)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            for (int i = 4; i < bindings.capacity(); i++) {
                bindings.get(i).binding(i)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }

            LongBuffer handle = stack.mallocLong(1);
            VkDescriptorSetLayoutCreateInfo layoutInfo =
                    VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(bindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(rt bloom)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "bloom descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSize = VkDescriptorPoolSize.calloc(2, stack);
            poolSize.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(4);
            poolSize.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(poolSize);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(rt bloom)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    descriptorPool, "bloom descriptor pool");

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(descriptorSetLayout));
            LongBuffer setHandle = stack.mallocLong(1);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandle),
                    "vkAllocateDescriptorSets(rt bloom)");
            long descriptorSet = setHandle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET,
                    descriptorSet, "bloom descriptor set");

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(BloomPushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(rt bloom)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    pipelineLayout, "bloom pipeline layout");

            long module = loadModule(vk, stack);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, "bloom shader module");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo =
                    VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            LongBuffer pipelineHandle = stack.mallocLong(1);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE,
                    pipelineInfo, null, pipelineHandle), "vkCreateComputePipelines(rt bloom)");
            VK10.vkDestroyShaderModule(vk, module, null);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE,
                    pipelineHandle.get(0), "bloom compute pipeline");

            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle),
                    "vkCreateSampler(rt bloom)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "bloom linear sampler");

            return new RtBloomPipeline(ctx, descriptorSetLayout, descriptorPool, descriptorSet,
                    pipelineLayout, pipelineHandle.get(0), sampler);
        }
    }

    public long sampler() {
        return sampler;
    }

    public void setImages(long rtView, long exposureView, long bloomAView, long bloomBView) {
        if (boundRtView == rtView && boundExposureView == exposureView
                && boundBloomAView == bloomAView && boundBloomBView == bloomBView) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long[] views = {rtView, exposureView, bloomAView, bloomBView};
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(6, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(6, stack);
            for (int i = 0; i < views.length; i++) {
                images.get(i).imageView(views[i]).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(i).sType$Default().dstSet(descriptorSet).dstBinding(i)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(i), 1));
            }
            images.get(4).imageView(bloomAView).sampler(sampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            images.get(5).imageView(bloomBView).sampler(sampler)
                    .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
            for (int i = 4; i < 6; i++) {
                writes.get(i).sType$Default().dstSet(descriptorSet).dstBinding(i)
                        .descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(i), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
        boundRtView = rtView;
        boundExposureView = exposureView;
        boundBloomAView = bloomAView;
        boundBloomBView = bloomBView;
    }

    public void dispatch(VkCommandBuffer cmd, int width, int height,
                         float threshold, float softKneeFraction, float radius) {
        float softKnee = threshold * softKneeFraction;
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "scene bloom")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, stack.longs(descriptorSet), null);
            for (int mode = 0; mode < 3; mode++) {
                ByteBuffer push = stack.malloc(BloomPushData.BYTE_SIZE);
                new BloomPushData(mode, threshold, softKnee, radius).write(push);
                VK10.vkCmdPushConstants(cmd, pipelineLayout,
                        VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
                VK10.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
                VulkanCommandEncoder.memoryBarrier(cmd, stack);
            }
        }
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack) {
        byte[] bytes;
        try (InputStream input = RtBloomPipeline.class.getResourceAsStream(SHADER)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module),
                    "vkCreateShaderModule(bloom)");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
