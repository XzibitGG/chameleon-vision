package com.chameleonvision.common.vision.pipeline;

import com.chameleonvision.common.vision.camera.CaptureStaticProperties;
import com.chameleonvision.common.vision.opencv.Contour;
import com.chameleonvision.common.vision.pipeline.pipe.*;
import com.chameleonvision.common.vision.pipeline.pipe.Collect2dTargetsPipe.Collect2dTargetsParams;
import com.chameleonvision.common.vision.pipeline.pipe.Draw2dCrosshairPipe.Draw2dCrosshairParams;
import com.chameleonvision.common.vision.pipeline.pipe.RotateImagePipe.RotateImageParams;
import com.chameleonvision.common.vision.target.PotentialTarget;
import com.chameleonvision.common.vision.target.TrackedTarget;
import edu.wpi.cscore.CameraServerCvJNI;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.wpi.cscore.VideoMode;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/** This class exists for the sole purpose of showing how pipes would interact in a pipeline */
public class DummyPipeline {
    private static ResizeImagePipe resizePipe = new ResizeImagePipe();
    private static RotateImagePipe rotatePipe = new RotateImagePipe();
    private static HSVPipe hsvPipe = new HSVPipe();
    private static FindContoursPipe findContoursPipe = new FindContoursPipe();
    private static Collect2dTargetsPipe collectTargetsPipe = new Collect2dTargetsPipe();
    private static Draw2dCrosshairPipe crosshairPipe = new Draw2dCrosshairPipe();

    private static List<Mat> outputMats = new ArrayList<>();

    public static void main(String[] args) {
        try {
            CameraServerCvJNI.forceLoad();
        } catch (UnsatisfiedLinkError | IOException e) {
            throw new RuntimeException("Failed to load JNI Libraries!");
        }

        // obviously not a useful test, purely for example.
        Mat fakeCameraMat = Imgcodecs.imread("D:\\chameleon-vision\\testimages\\2020\\RedLoading-030in-Down.jpg");

        PipeResult<Mat> resizeResult = resizePipe.apply(fakeCameraMat);

        rotatePipe.setParams(new RotateImageParams(RotateImageParams.ImageRotation.DEG_90));
        PipeResult<Mat> rotateResult = rotatePipe.apply(resizeResult.result);
        outputMats.add(rotateResult.result);

        hsvPipe.setParams(new HSVPipe.HSVParams(new Scalar(0, 0, 100), new Scalar(255, 200, 255)));
        Mat hsvPipeInputMat = new Mat();
        Imgproc.cvtColor(fakeCameraMat, hsvPipeInputMat, Imgproc.COLOR_BGR2HSV);
        PipeResult<Mat> hsvPipeResult = hsvPipe.apply(hsvPipeInputMat);
        Mat hsvPipeOutputMat = new Mat();
        Imgproc.cvtColor(hsvPipeResult.result, hsvPipeOutputMat, Imgproc.COLOR_GRAY2BGR);

        Mat input = hsvPipeResult.result;
        PipeResult<List<Contour>> findContoursPipeResult = findContoursPipe.apply(input);
        Collections.sort(findContoursPipeResult.result, Comparator.comparing(Contour::getArea).reversed());
        Mat contourOutputMat = hsvPipeOutputMat;
        List<MatOfPoint> contours = new ArrayList<>();
        findContoursPipeResult.result.forEach((contour) -> contours.add(contour.mat));
        Imgproc.drawContours(contourOutputMat, contours, 0, new Scalar(255), 4);
        outputMats.add(contourOutputMat);


        PipeResult<List<TrackedTarget>> collectTargetsResult = collectTargetsPipe.apply(List.of(new PotentialTarget(findContoursPipeResult.result.get(0))));

        crosshairPipe.setParams(new Draw2dCrosshairParams());
        PipeResult<Mat> crosshairPipeResult = crosshairPipe.apply(Pair.of(contourOutputMat, collectTargetsResult.result));
        outputMats.add(crosshairPipeResult.result);


        long fullTime = resizeResult.nanosElapsed + rotateResult.nanosElapsed;
        System.out.println(fullTime / 1.0e+6 + "ms elapsed");
        showOutputs();
    }

    public static void showOutputs(){
        Mat outputMat = new Mat();
        Core.hconcat(outputMats, outputMat);
        HighGui.imshow("Outputs", outputMat);
        HighGui.waitKey();
    }
}
