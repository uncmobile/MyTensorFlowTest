package edu.unc.nirjon.mytensorflowtest;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            foo();
        }catch (Exception ex)
        {
            Log.v("MYTAG", "" + ex.toString());
        }
    }

    void foo() throws Exception
    {
        /*Step 1 of 3: Prepare the Interpreter*/
        AssetFileDescriptor afd = getAssets().openFd("mobilenet_v1_1.0_224_quant.tflite");
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel fc = fis.getChannel();
        MappedByteBuffer mbb = fc.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getLength());
        Interpreter interpreter = new Interpreter(mbb);

        /*Step 2 of 3: Prepare Input and Output*/
        Drawable drawable = getResources().getDrawable(R.drawable.bee);
        Bitmap bitmap = ((BitmapDrawable)drawable).getBitmap(); //manually shaped to: 224 x 224 (32 bit)
        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
        TensorBuffer result = TensorBuffer.createFixedSize(interpreter.getOutputTensor(0).shape(),
                interpreter.getOutputTensor(0).dataType());

        /*Step 3 of 3: Run the Interpreter and Print Result.*/
        interpreter.run(tensorImage.getBuffer(), result.getBuffer().rewind());

        ByteBuffer bbuff = result.getBuffer();
        for(int i = 0; i < bbuff.capacity(); i++) {
            if((bbuff.get(i) & 0xFF) > 0){
                Log.v("MYTAG", "Class: " + i + ", Probability: " + (bbuff.get(i) & 0xFF)/255f);
            }
        }


    }
}
