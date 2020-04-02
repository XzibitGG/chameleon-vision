package com.chameleonvision.common.vision.pipeline;

import com.chameleonvision.common.vision.camera.CaptureStaticProperties;
import com.chameleonvision.common.vision.opencv.Contour;
import com.chameleonvision.common.vision.pipeline.pipe.*;
import com.chameleonvision.common.vision.pipeline.pipe.Collect2dTargetsPipe.Collect2dTargetsParams;
import com.chameleonvision.common.vision.pipeline.pipe.Draw2dCrosshairPipe.Draw2dCrosshairParams;
import com.chameleonvision.common.vision.pipeline.pipe.ErodeDilatePipe.ErodeDilateParams;
import com.chameleonvision.common.vision.pipeline.pipe.ResizeImagePipe.ResizeImageParams;
import com.chameleonvision.common.vision.pipeline.pipe.RotateImagePipe.RotateImageParams;
import com.chameleonvision.common.vision.target.PotentialTarget;
import com.chameleonvision.common.vision.target.TrackedTarget;
import edu.wpi.cscore.CameraServerCvJNI;

import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.wpi.cscore.VideoMode;
import edu.wpi.first.wpilibj.geometry.Rotation2d;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.opencv.core.*;
import org.opencv.core.Point;
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
    private static ErodeDilatePipe erodeDilatePipe = new ErodeDilatePipe();


    private static List<Mat> outputMats = new ArrayList<>();

    public static void main(String[] args) {
        try {
            CameraServerCvJNI.forceLoad();
        } catch (UnsatisfiedLinkError | IOException e) {
            throw new RuntimeException("Failed to load JNI Libraries!");
        }

        // obviously not a useful test, purely for example.
        Mat fakeCameraMat = Imgcodecs.imread("D:\\chameleon-vision\\testimages\\2020\\RedLoading-030in-Down.jpg");

        resizePipe.setParams(new ResizeImageParams(640, 480));
        PipeResult<Mat> resizeResult = resizePipe.apply(fakeCameraMat);
        outputMats.add(resizeResult.result);

        erodeDilatePipe.setParams(new ErodeDilateParams(true, true, 20));
        PipeResult<Mat> erodeDilateResult= erodeDilatePipe.apply(resizeResult.result);
        outputMats.add(erodeDilateResult.result);

        rotatePipe.setParams(new RotateImageParams(RotateImageParams.ImageRotation.DEG_90));
        PipeResult<Mat> rotateResult = rotatePipe.apply(resizeResult.result);
        outputMats.add(rotateResult.result);

        hsvPipe.setParams(new HSVPipe.HSVParams(new Scalar(0, 0, 100), new Scalar(255, 200, 255)));
        Mat hsvPipeInputMat = new Mat();
        Imgproc.cvtColor(fakeCameraMat, hsvPipeInputMat, Imgproc.COLOR_BGR2HSV);
        PipeResult<Mat> hsvPipeResult = hsvPipe.apply(hsvPipeInputMat);
        Mat hsvPipeOutputMat = new Mat();
        Imgproc.cvtColor(hsvPipeResult.result, hsvPipeOutputMat, Imgproc.COLOR_GRAY2BGR);
        outputMats.add(hsvPipeOutputMat);

        Mat input = hsvPipeResult.result;
        PipeResult<List<Contour>> findContoursPipeResult = findContoursPipe.apply(input);
        Collections.sort(findContoursPipeResult.result, Comparator.comparing(Contour::getArea).reversed());
        Mat contourOutputMat = hsvPipeOutputMat;
        List<MatOfPoint> contours = new ArrayList<>();
        findContoursPipeResult.result.forEach((contour) -> contours.add(contour.mat));
        Imgproc.drawContours(contourOutputMat, contours, 0, new Scalar(255), 4);
        outputMats.add(contourOutputMat);

        collectTargetsPipe.setParams(new Collect2dTargetsParams(
            new CaptureStaticProperties(
                    new VideoMode(1, 640, 480, 60),
                    68.5
            ),
             TrackedTarget.RobotOffsetPointMode.Single,
             60, 60,
             new Point(320, 240),
             TrackedTarget.TargetOffsetPointRegion.Center,
             TrackedTarget.TargetOrientation.Landscape
        ));
        PipeResult<List<TrackedTarget>> collectTargetsResult = collectTargetsPipe.apply(List.of(new PotentialTarget(findContoursPipeResult.result.get(0))));

        crosshairPipe.setParams(new Draw2dCrosshairParams(
                TrackedTarget.RobotOffsetPointMode.Single,
                new Point(0, 0),
                true,
                new Color(255, 0, 255)
        ));
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
