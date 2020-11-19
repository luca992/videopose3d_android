package com.example.videopose3d;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.SensorEventListener;
import android.location.LocationListener;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.jni.NdkHelper;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.IValue;
import org.tensorflow.lite.examples.noah.lib.Device;
import org.tensorflow.lite.examples.noah.lib.KeyPoint;
import org.tensorflow.lite.examples.noah.lib.Person;
import org.tensorflow.lite.examples.noah.lib.Posenet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.sql.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends AppCompatActivity implements GLSurfaceView.Renderer, CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {
    public final String TAG = "MainAct";

    ImageView imgDealed;

    LinearLayout linear;

    String vocPath, calibrationPath;

    Posenet posenet;

    private static final int INIT_FINISHED = 0x00010001;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    private final int CONTEXT_CLIENT_VERSION = 3;
    private final long[] shape = {2, 272, 17, 2};

    //OpenGL SurfaceView
    private GLSurfaceView mGLSurfaceView;
    private Module module = null;

    private int frameNum = 0;

    private float[][][] keypts2d = {};

    private FloatBuffer mInputTensorBuffer = null;
    private Tensor mInputTensor = null;

    //load up native C code
    static {
        System.loadLibrary("drawer");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);//will hide the title.
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getSupportActionBar().hide(); //hide the title bar.

        //set the view to main act view
        setContentView(R.layout.activity_main);

        posenet = new Posenet(this, "posenet_model.tflite", Device.NNAPI);

        imgDealed = (ImageView) findViewById(R.id.img_dealed);

        //mIsJavaCamera is bool describing whether or not we're using JavaCameraView. Which we always are, it seems.

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_cam_view);


        //check the the view was found by ID and all is well
        if (mOpenCvCameraView == null) {
            Log.e(TAG, "mOpenCvCameraView came up null");
        }
        else {
            Log.d(TAG, "mOpenCvCameraView non-null, OK");
        }

        //make our OpenCvCameraView visible and set the listener for the camera
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //instantiate new GLSurfaceView
        mGLSurfaceView = new GLSurfaceView(this);
        linear = (LinearLayout) findViewById(R.id.surfaceLinear);

        //mGLSurfaceView.setEGLContextClientVersion(CONTEXT_CLIENT_VERSION);

        //set the renderer for the GLSurfaceView
        mGLSurfaceView.setRenderer(this);

        linear.addView(mGLSurfaceView, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT));


        Bitmap bitmap = null;


        try {
            //Load model: loading serialized torchscript module from packaged into app android asset model.pt,
            module = Module.load(assetFilePath(this, "processed_mod.pt"));
        }

        catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }
        

        Log.i("DBUG", "Read in VideoPose3D successfully");

        /*
        //Preparing input tensor from the image (in torchvision format)
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //inputTensor’s shape is 1x3xHxW, where H and W are bitmap height and width appropriately.


        //Running the model - run loaded module’s forward method, get result as org.pytorch.Tensor outputTensor with shape 1x1000
        assert module != null;
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();


        //Get output tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray(); //returns java array of floats with scores for every image net class
         */
    }

    //use this OpenCV loader callback to instantiate Mat objects, otherwise we'll get an error about Mat not being found
    public BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            Log.i(TAG, "BaseLoaderCallback called!");

            if (status == LoaderCallbackInterface.SUCCESS) {//instantiate everything we need from OpenCV
                //everything succeeded
                Log.i(TAG, "OpenCV loaded successfully, everything created");
                mOpenCvCameraView.enableView();
            }

            else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onResume() {
        super.onResume();

        mGLSurfaceView.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        }

        else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        mGLSurfaceView.onPause();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    //input (N, 17, 2) return (N, 17, 3)
    public float[][] interface3d(Module mod, float[][][] kpts, int width, int height) {
        long start = System.currentTimeMillis();

        //do some correction of array format of kpts

        //float[][][] keypoints = new float[1][][];
        //keypoints[0] = kpts;

        //normalize coordinates
        Camera.normalize_screen_coordinates(kpts, 1000, 1002);

        UnchunkedGenerator gen = new UnchunkedGenerator(kpts, Common.pad, Common.causal_shift, true,
                Common.kps_left, Common.kps_right, Common.joints_left, Common.joints_right);

        float[][] prediction = evaluate(gen, mod, true);

        prediction = Camera.camera_to_world(prediction, Common.rot, 0);

        //min out prediction


        long end = System.currentTimeMillis();
        Log.i(TAG, String.format("interface3d took total %d ms", end - start));

        return prediction;
    }



    public float[] flatten(float[][][][] input, long[] shape) {
        long needed = 1;

        long start = System.currentTimeMillis();

        for (int i = 0; i < shape.length; i++) {
            needed *= shape[i];
        }

        //Log.i(TAG, String.format("We need %d for ret", needed));

        float[] ret = new float[(int) needed];

        //Log.i(TAG, String.format("First dim is %d", input.length));
        //Log.i(TAG, String.format("Second dim is %d", input[0].length));


        int head = 0;

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    for (int l = 0; l < shape[3]; l++) {
                        ret[head++] = input[i][j][k][l];
                    }
                }
            }
        }

        long end = System.currentTimeMillis();

        Log.i(TAG, String.format("flatten took %d ms", end-start));

        return ret;
    }


    /*
    for _, batch, batch_2d in test_generator.next_epoch():
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        #Run the positional model
        predicted_3d_pos = model_pos(inputs_2d)


        if test_generator.augment_enabled():


            # Undo flipping and take average with non-flipped version
            predicted_3d_pos[1, :, :, 0] *= -1
            predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
            predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)


        if return_predictions:
            return predicted_3d_pos.squeeze(0).cpu().numpy()*/


    public float[][] evaluate(UnchunkedGenerator gen, Module mod, boolean return_predictions) {
        long start = System.currentTimeMillis();

        float[][][][] batch_2d = gen.next_epoch();

        float[] batch_2d_flat = flatten(batch_2d, shape);
        //Log.i(TAG, String.format("Length of batch_2d_flat is %d", batch_2d_flat.length));

        if (mInputTensorBuffer==null) {
            //allocate the memory for input tensor data once
            mInputTensorBuffer = Tensor.allocateFloatBuffer(batch_2d_flat.length);
            mInputTensor = Tensor.fromBlob(mInputTensorBuffer, shape);
        }

        //put data into input tensor
        mInputTensorBuffer.clear();
        mInputTensorBuffer.put(batch_2d_flat);

        //Tensor inputTensor = Tensor.fromBlob(batch_2d_flat, shape);

        Tensor predicted_3d_pos = mod.forward(IValue.from(mInputTensor)).toTensor();

        float[] result = predicted_3d_pos.getDataAsFloatArray();


        /*
        long[] shape = predicted_3d_pos.shape();

        for (long l : shape) {
            Log.i(TAG, String.format("Shape: %d", (int)l));
        }

        //Log.i(TAG, String.format("Result is length %d", result.length));*/

        long end = System.currentTimeMillis();
        Log.i(TAG, String.format("evaluate took total %d ms", end - start));

        return null;
    }


    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        //Looking for file [files directory]/assetname - here files directory will be "assets"
        File file = new File(context.getFilesDir(), assetName);
        Log.i("DBUG", file.getAbsolutePath());

        //Default: return absolute path of file
        if (file.exists() && file.length() > 0) {
            Log.i("DBUG", "Found specified file, returning abs path");
            return file.getAbsolutePath();
        }


        Log.i("DBUG", "Specified file doesn't exist or was empty");
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {
        //initialize the OpenGL ES framework
        Log.i(TAG, "Initializing the GL ES framework/starting drawing...");
        NdkHelper.glesInit();
    }

    @Override
    public void onSurfaceChanged(GL10 gl10, int i, int i1) {

    }

    @Override
    public void onDrawFrame(GL10 gl10) {
        //Log.i(TAG, "Rendering the frame...");
        NdkHelper.glesRender();
    }

    @Override
    public void onClick(View view) {

    }


    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        long start = System.currentTimeMillis();

        //get the frame as a mat in RGBA format
        Mat im = inputFrame.rgba();


        if (frameNum % 2 != 1) {


            Mat im_resized = new Mat();

            Imgproc.resize(im, im_resized, new Size(257, 257));

            Log.i(TAG, String.format("Image is %d x %d", im.width(), im.height()));

            Bitmap bitmap = Bitmap.createBitmap(257, 257, Bitmap.Config.ARGB_8888);

            org.opencv.android.Utils.matToBitmap(im_resized, bitmap);

            Log.i(TAG, String.format("Bitmap is %d x %d", bitmap.getWidth(), bitmap.getHeight()));

            float htRatio = 480f / 257f;
            float wdRatio = 720f / 257f;

            //Bitmap croppedBitmap = cropBitmap(bitmap);

            //Created scaled version of bitmap for model input (scales it to 257 x 257)
            //Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, 257, 257, true);

            //Perform inference
            Person person = posenet.estimateSinglePose(bitmap);

            float[][] kpts = new float[17][2];


            List<KeyPoint> result = person.getKeyPoints();

            for (int i = 0; i < result.size(); i++) {
                KeyPoint thisKpt = result.get(i);

                float x = thisKpt.getPosition().getX() * wdRatio;
                float y = thisKpt.getPosition().getY() * htRatio;


                if (thisKpt.getScore() > 0.8) {

                    //draw this point on the preview
                    Imgproc.circle(im, new Point((int) x, (int) y), 2, new Scalar(0, 0, 255), 2);
                }

                kpts[i][0] = x;
                kpts[i][1] = y;

            }

            if (frameNum == 0) {
                keypts2d = Arrays.copyOf(keypts2d, 30);

                for (int i = 0; i < 30; i++) {
                    keypts2d[i] = kpts;
                }
            }

            else {
                //do a pop
                for (int i = 0; i < 29; i++) {
                    keypts2d[i] = keypts2d[i + 1];
                }

                //append the current frame
                keypts2d[29] = kpts;
            }

            interface3d(module, keypts2d, 720, 480);
        }

        frameNum++;

        long end = System.currentTimeMillis();

        Log.i(TAG, String.format("onCameraFrame took total %d ms", end-start));

        //whatever gets returned here is what's displayed
        return im;
    }

    /*
    public String type2str(int type) {
        String r;

         depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }*/

    /** Crop Bitmap to maintain aspect ratio of model input. */
    private Bitmap cropBitmap(Bitmap bitmap) {
        float bitmapRatio = (float)bitmap.getHeight() / bitmap.getWidth();

        float modelInputRatio = 1.0f;

        //first set new edited bitmap equal to the passed one
        Bitmap croppedBitmap = bitmap;

        //Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
        double maxDifference = 1e-5;

        //Checks if the bitmap has similar aspect ratio as the required model input.
        if (Math.abs(modelInputRatio - bitmapRatio) < maxDifference) {
            return croppedBitmap;
        }

        else if (modelInputRatio < bitmapRatio) {
            //New image is taller so we are height constrained.
            float cropHeight = bitmap.getHeight() - ((float)bitmap.getWidth() / modelInputRatio);

            croppedBitmap = Bitmap.createBitmap(bitmap, 0, (int)(cropHeight / 2), bitmap.getWidth(), (int)(bitmap.getHeight() - cropHeight));
        }

        else {
            //Log.i(TAG, "Cropping...");

            float cropWidth = bitmap.getWidth() - ((float)bitmap.getHeight() * modelInputRatio); //=720-480 = 240


            croppedBitmap = Bitmap.createBitmap(bitmap, (int)(cropWidth / 2), 0, (int)(bitmap.getWidth() - cropWidth), bitmap.getHeight());
        }

        return croppedBitmap;
    }

}