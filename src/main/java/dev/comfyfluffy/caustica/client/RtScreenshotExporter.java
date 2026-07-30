package dev.comfyfluffy.caustica.client;

import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtComposite;
import net.minecraft.ChatFormatting;
import net.minecraft.network.chat.ClickEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.util.Util;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

/** Adds a residual-exposed scene-linear ACEScg EXR beside vanilla's ordinary F2 PNG. */
public final class RtScreenshotExporter {
    private RtScreenshotExporter() {
    }

    public static void export(File workDir, Consumer<Component> callback) {
        Path screenshotDirectory = workDir.toPath().resolve("screenshots");
        Path output = nextPairedPath(screenshotDirectory);
        try {
            if (!RtComposite.INSTANCE.exportLatestResidualExposureExr(output)) {
                return;
            }
            File file = output.toFile().getAbsoluteFile();
            Component link = Component.literal(file.getName())
                    .withStyle(ChatFormatting.UNDERLINE)
                    .withStyle(style -> style.withClickEvent(new ClickEvent.OpenFile(file)));
            callback.accept(Component.literal("Saved residual-exposure ACEScg EXR: ").append(link));
            CausticaMod.LOGGER.info("Saved residual-exposure ACEScg screenshot to {}", file);
        } catch (Exception e) {
            CausticaMod.LOGGER.warn("Couldn't save residual-exposure ACEScg screenshot", e);
            callback.accept(Component.literal("Couldn't save Caustica EXR: " + e.getMessage())
                    .withStyle(ChatFormatting.RED));
        }
    }

    private static Path nextPairedPath(Path directory) {
        String base = Util.getFilenameFormattedDateTime();
        int count = 1;
        while (true) {
            String suffix = count == 1 ? "" : "_" + count;
            Path candidate = directory.resolve(base + suffix + ".exr");
            Path vanillaPng = directory.resolve(base + suffix + ".png");
            if (!Files.exists(candidate) && !Files.exists(vanillaPng)) {
                return candidate;
            }
            count++;
        }
    }
}
