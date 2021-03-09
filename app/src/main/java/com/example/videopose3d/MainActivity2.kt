package com.example.videopose3d

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.*
import android.widget.ImageView
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.jni.NdkHelper
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.android.*
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.tensorflow.lite.examples.noah.lib.Device
import org.tensorflow.lite.examples.noah.lib.Posenet
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.util.*
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity2 : AppCompatActivity(), GLSurfaceView.Renderer, CvCameraViewListener2, View.OnClickListener {
    val TAG = "MainAct"
    var imgDealed: ImageView? = null
    var linear: LinearLayout? = null
    var vocPath: String? = null
    var calibrationPath: String? = null
    var posenet: Posenet? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private val mIsJavaCamera = true
    private val mItemSwitchCamera: MenuItem? = null
    private val CONTEXT_CLIENT_VERSION = 3
    private val shape = longArrayOf(2, 272, 17, 2)

    //OpenGL SurfaceView
    private var mGLSurfaceView: GLSurfaceView? = null
    private var module: Module? = null
    private var frameNum = 0
    private var keypts2d = arrayOf<Array<FloatArray>>()
    private var mInputTensorBuffer: FloatBuffer? = null
    private var mInputTensor: Tensor? = null

    companion object {
        private const val INIT_FINISHED = 0x00010001

        /**
         * Copies specified asset to the file in /files app directory and returns this file absolute path.
         *
         * @return absolute file path
         */
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String?): String {
            //Looking for file [files directory]/assetname - here files directory will be "assets"
            val file = File(context.filesDir, assetName)
            Log.i("DBUG", file.absolutePath)

            //Default: return absolute path of file
            if (file.exists() && file.length() > 0) {
                Log.i("DBUG", "Found specified file, returning abs path")
                return file.absolutePath
            }
            Log.i("DBUG", "Specified file doesn't exist or was empty")
            context.assets.open(assetName!!).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }

        //load up native C code
        init {
            System.loadLibrary("drawer")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE) //will hide the title.
        window.setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN)
        supportActionBar!!.hide() //hide the title bar.

        //set the view to main act view
        setContentView(R.layout.activity_main)
        posenet = Posenet(this, "posenet_model.tflite", Device.NNAPI)
        imgDealed = findViewById<View>(R.id.img_dealed) as ImageView

        //mIsJavaCamera is bool describing whether or not we're using JavaCameraView. Which we always are, it seems.
        mOpenCvCameraView = findViewById<View>(R.id.java_cam_view) as CameraBridgeViewBase


        //check the the view was found by ID and all is well
        if (mOpenCvCameraView == null) {
            Log.e(TAG, "mOpenCvCameraView came up null")
        } else {
            Log.d(TAG, "mOpenCvCameraView non-null, OK")
        }

        //make our OpenCvCameraView visible and set the listener for the camera
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)

        //instantiate new GLSurfaceView
        mGLSurfaceView = GLSurfaceView(this)
        linear = findViewById<View>(R.id.surfaceLinear) as LinearLayout

        //mGLSurfaceView.setEGLContextClientVersion(CONTEXT_CLIENT_VERSION);

        //set the renderer for the GLSurfaceView
        mGLSurfaceView!!.setRenderer(this)
        linear!!.addView(mGLSurfaceView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT))
        val bitmap: Bitmap? = null
        try {
            //Load model: loading serialized torchscript module from packaged into app android asset model.pt,
            module = Module.load(assetFilePath(this, "processed_mod.pt"))
        } catch (e: IOException) {
            Log.e("PytorchHelloWorld", "Error reading assets", e)
            finish()
        }
        Log.i("DBUG", "Read in VideoPose3D successfully")

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
    var mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            Log.i(TAG, "BaseLoaderCallback called!")
            if (status == SUCCESS) { //instantiate everything we need from OpenCV
                //everything succeeded
                Log.i(TAG, "OpenCV loaded successfully, everything created")
                mOpenCvCameraView!!.enableView()
            } else {
                super.onManagerConnected(status)
            }
        }
    }

    override fun onStart() {
        super.onStart()
    }

    override fun onResume() {
        super.onResume()
        mGLSurfaceView!!.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onPause() {
        super.onPause()
        mGLSurfaceView!!.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    //input (N, 17, 2) return (N, 17, 3)
    fun interface3d(mod: Module?, kpts: Array<Array<FloatArray>>?, width: Int, height: Int): Array<FloatArray?>? {
        val start = System.currentTimeMillis()

        //do some correction of array format of kpts

        //float[][][] keypoints = new float[1][][];
        //keypoints[0] = kpts;

        //normalize coordinates
        Camera.normalize_screen_coordinates(kpts, 1000, 1002)
        val gen = UnchunkedGenerator(kpts, Common.pad, Common.causal_shift, true,
                Common.kps_left, Common.kps_right, Common.joints_left, Common.joints_right)
        var prediction = evaluate(gen, mod, true)
        prediction = Camera.camera_to_world(prediction, Common.rot, 0)

        //min out prediction
        val end = System.currentTimeMillis()
        Log.i(TAG, String.format("interface3d took total %d ms", end - start))
        return prediction
    }

    fun flatten(input: Array<Array<Array<FloatArray>>>, shape: LongArray): FloatArray {
        var needed: Long = 1
        val start = System.currentTimeMillis()
        for (i in shape.indices) {
            needed *= shape[i]
        }

        //Log.i(TAG, String.format("We need %d for ret", needed));
        val ret = FloatArray(needed.toInt())

        //Log.i(TAG, String.format("First dim is %d", input.length));
        //Log.i(TAG, String.format("Second dim is %d", input[0].length));
        var head = 0
        for (i in 0 until shape[0]) {
            for (j in 0 until shape[1]) {
                for (k in 0 until shape[2]) {
                    for (l in 0 until shape[3]) {
                        ret[head++] = input[i.toInt()][j.toInt()][k.toInt()][l.toInt()]
                    }
                }
            }
        }
        val end = System.currentTimeMillis()
        Log.i(TAG, String.format("flatten took %d ms", end - start))
        return ret
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
    fun evaluate(gen: UnchunkedGenerator, mod: Module?, return_predictions: Boolean): Array<FloatArray?>? {
        val start = System.currentTimeMillis()
        val batch_2d = gen.next_epoch()
        val batch_2d_flat = flatten(batch_2d, shape)
        //Log.i(TAG, String.format("Length of batch_2d_flat is %d", batch_2d_flat.length));
        if (mInputTensorBuffer == null) {
            //allocate the memory for input tensor data once
            mInputTensorBuffer = Tensor.allocateFloatBuffer(batch_2d_flat.size)
            mInputTensor = Tensor.fromBlob(mInputTensorBuffer, shape)
        }

        //put data into input tensor
        mInputTensorBuffer!!.clear()
        mInputTensorBuffer!!.put(batch_2d_flat)

        //Tensor inputTensor = Tensor.fromBlob(batch_2d_flat, shape);
        val predicted_3d_pos = mod!!.forward(IValue.from(mInputTensor)).toTensor()
        val result = predicted_3d_pos.dataAsFloatArray



        val shape = predicted_3d_pos.shape()

        Log.i(TAG, "Shape: ${result}");


        //Log.i(TAG, "Result is length ${result}")
        val end = System.currentTimeMillis()
        Log.i(TAG, String.format("evaluate took total %d ms", end - start))
        return null
    }

    override fun onSurfaceCreated(gl10: GL10, eglConfig: EGLConfig) {
        //initialize the OpenGL ES framework
        Log.i(TAG, "Initializing the GL ES framework/starting drawing...")
        NdkHelper.glesInit()
    }

    override fun onSurfaceChanged(gl10: GL10, i: Int, i1: Int) {}
    override fun onDrawFrame(gl10: GL10) {
        //Log.i(TAG, "Rendering the frame...");
        NdkHelper.glesRender()
    }

    override fun onClick(view: View) {}
    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}
    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val start = System.currentTimeMillis()

        //get the frame as a mat in RGBA format
        val im = inputFrame.rgba()
        if (frameNum % 2 != 1) {
            val im_resized = Mat()
            Imgproc.resize(im, im_resized, Size(257.toDouble(), 257.toDouble()))
            Log.i(TAG, String.format("Image is %d x %d", im.width(), im.height()))
            val bitmap = Bitmap.createBitmap(257, 257, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(im_resized, bitmap)
            Log.i(TAG, String.format("Bitmap is %d x %d", bitmap.width, bitmap.height))
            val htRatio = 480f / 257f
            val wdRatio = 720f / 257f

            //Bitmap croppedBitmap = cropBitmap(bitmap);

            //Created scaled version of bitmap for model input (scales it to 257 x 257)
            //Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, 257, 257, true);

            //Perform inference
            val person = posenet!!.estimateSinglePose(bitmap)
            val kpts = Array(17) { FloatArray(2) }
            val result = person.keyPoints
            for (i in result.indices) {
                val thisKpt = result[i]
                val x = thisKpt.position.x * wdRatio
                val y = thisKpt.position.y * htRatio
                if (thisKpt.score > 0.8) {

                    //draw this point on the preview
                    Imgproc.circle(im, Point(x.toDouble(), y.toDouble()), 2, Scalar(0.toDouble(), 0.toDouble(), 255.toDouble()), 2)
                }
                kpts[i][0] = x
                kpts[i][1] = y
            }
            if (frameNum == 0) {
                keypts2d = Arrays.copyOf(keypts2d, 30)
                for (i in 0..29) {
                    keypts2d[i] = kpts
                }
            } else {
                //do a pop
                for (i in 0..28) {
                    keypts2d[i] = keypts2d[i + 1]
                }

                //append the current frame
                keypts2d[29] = kpts
            }
            lifecycleScope.launch(Dispatchers.Default) {
                interface3d(module, keypts2d, 720, 480)
            }
        }
        frameNum++
        val end = System.currentTimeMillis()
        Log.i(TAG, String.format("onCameraFrame took total %d ms", end - start))

        //whatever gets returned here is what's displayed
        return im
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
    /** Crop Bitmap to maintain aspect ratio of model input.  */
    private fun cropBitmap(bitmap: Bitmap): Bitmap {
        val bitmapRatio = bitmap.height.toFloat() / bitmap.width
        val modelInputRatio = 1.0f

        //first set new edited bitmap equal to the passed one
        var croppedBitmap = bitmap

        //Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
        val maxDifference = 1e-5

        //Checks if the bitmap has similar aspect ratio as the required model input.
        croppedBitmap = if (Math.abs(modelInputRatio - bitmapRatio) < maxDifference) {
            return croppedBitmap
        } else if (modelInputRatio < bitmapRatio) {
            //New image is taller so we are height constrained.
            val cropHeight = bitmap.height - bitmap.width.toFloat() / modelInputRatio
            Bitmap.createBitmap(bitmap, 0, (cropHeight / 2).toInt(), bitmap.width, (bitmap.height - cropHeight).toInt())
        } else {
            //Log.i(TAG, "Cropping...");
            val cropWidth = bitmap.width - bitmap.height.toFloat() * modelInputRatio //=720-480 = 240
            Bitmap.createBitmap(bitmap, (cropWidth / 2).toInt(), 0, (bitmap.width - cropWidth).toInt(), bitmap.height)
        }
        return croppedBitmap
    }
}