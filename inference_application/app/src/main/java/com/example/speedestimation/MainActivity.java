package com.example.speedestimation;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.nfc.Tag;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity {

    //Using the Accelometer & Gyroscoper
    private SensorManager mSensorManager = null;

    //Using the Gyroscope
    private SensorEventListener mGyroLis;
    private Sensor mGgyroSensor = null;

    //Using the Accelometer
    private SensorEventListener mAccLis;
    private Sensor mAccelerometerSensor = null;

    //Roll and Pitch
    private double pitch;
    private double roll;
    private double yaw;

    //timestamp and dt
    private double timestamp;
    private double dt;
    private double RAD2DGR = 180 / Math.PI;
    private static final float NS2S = 1.0f/1000000000.0f;
    private double X,Y,Z;
    private double speed, steering;
    Timer timer;
    private List<buf> buffer= new ArrayList<buf>();
    private boolean isStarting = false;
    private TextView t_X, t_Y, t_Z, t_Pitch, t_Roll, t_Yaw, t_speed, t_steering;
    private BackKeyClickHandler backKeyClickHandler;
    private String n_speed = "0";
    private String n_steering = "0";

    float[][] input;
    float[][] output;
    Interpreter model;

    private static final String TAG = "MyActivity";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 화면 안꺼짐
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        //Remove title bar
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //Remove notification bar
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        ApplicationClass.setContext(getResources());
        ApplicationClass.setcnxt(this);

        backKeyClickHandler = new BackKeyClickHandler(this);

        //Using the Gyroscope & Accelometer
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        //Using the gyroscope
        mGgyroSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mGyroLis = new GyroscopeListener();

        //Using the Accelerometer
        mAccelerometerSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mAccLis = new AccelerometerListener();


        //Get text id
        t_X = (TextView) findViewById(R.id.x);
        t_Y = (TextView) findViewById(R.id.y);
        t_Z = (TextView) findViewById(R.id.z);
        t_Pitch = (TextView) findViewById(R.id.pitch);
        t_Roll = (TextView) findViewById(R.id.roll);
        t_Yaw = (TextView) findViewById(R.id.yaw);

        t_speed = (TextView)findViewById(R.id.speed);
        t_steering = (TextView)findViewById(R.id.steering);

        //init parameter
        pitch = 0;
        roll = 0;
        yaw = 0;
        X = 0;
        Y = 0;
        Z = 0;
        speed = 0;
        steering = 0;
        buffer.add(new buf(0,0,0,0,0,0,0,0));

        // make interpreter for tflite
        model = getTfliteInterpreter("model.tflite");

        input = new float[16][6];
        output = new float[1][1];

        final Button start_stop = (Button) findViewById(R.id.start_stop);
        start_stop.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 시작 버튼
                if(!isStarting){
                    isStarting = true;

                    //Start listening sensor data
                    mSensorManager.registerListener(mGyroLis, mGgyroSensor, SensorManager.SENSOR_DELAY_UI);
                    mSensorManager.registerListener(mAccLis, mAccelerometerSensor, SensorManager.SENSOR_DELAY_UI);

                    //버튼 색상 및 텍스트 변경, green->red, start->stop
                    start_stop.setBackgroundResource(R.drawable.button_style_2);
                    start_stop.setText("Stop");

                    // 데이터를 0.25초에 한번씩 기록
                    getSensorData GSD = new getSensorData();
                    timer = new Timer();
                    timer.schedule(GSD, 0, 250);
                }
                else{
                    isStarting = false;

                    mSensorManager.unregisterListener(mGyroLis);
                    mSensorManager.unregisterListener(mAccLis);

                    start_stop.setBackgroundResource(R.drawable.button_style_1);
                    start_stop.setText("Start");

                    //기록 중지
                    timer.cancel();

                    //변수 초기화
                    buffer.clear();
                    buffer.add(new buf(0,0,0,0,0,0, 0, 0));

                    t_Pitch.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).pitch));
                    t_Roll.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).roll));
                    t_Yaw.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).yaw));
                    t_X.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).x));
                    t_Y.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).y));
                    t_Z.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).z));
                }
            }
        });
    }



    @Override public void onBackPressed() {
        backKeyClickHandler.onBackPressed();
    }


    final Handler handler = new Handler(){
        public void handleMessage(Message msg){
            //activity_main의 text 설정
            t_Pitch.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).pitch));
            t_Roll.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).roll));
            t_Yaw.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).yaw));
            t_X.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).x));
            t_Y.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).y));
            t_Z.setText(String.format("%.4f  ", buffer.get(buffer.size() -1).z));
            t_speed.setText(String.format("%.1f  ", buffer.get(buffer.size() -1).speed));
            t_steering.setText(String.format("%.0f  ", buffer.get(buffer.size() -1).steering));
        }
    };

    class getSensorData extends TimerTask{
        @Override
        public void run() {
            // buffer의 내용으로 textview 설정
            Message msg = handler.obtainMessage();
            handler.sendMessage(msg);


            // set buffer
            if(buffer.size() < 16)
                buffer.add(new buf((float)(pitch*RAD2DGR), (float)(roll*RAD2DGR), (float)(yaw*RAD2DGR), (float)X, (float)Y, (float)Z, (float)speed, (float) steering));
            else{
                buffer.remove(0);
                buffer.add(new buf((float)(pitch*RAD2DGR), (float)(roll*RAD2DGR), (float)(yaw*RAD2DGR), (float)X, (float)Y, (float)Z, (float)speed, (float) steering));


                for(int i=0;i<16;i++){
                    input[i][0] = buffer.get(i).x;
                    input[i][1] = buffer.get(i).y;
                    input[i][2] = buffer.get(i).z;
                    input[i][3] = buffer.get(i).pitch;
                    input[i][4] = buffer.get(i).roll;
                    input[i][5] = buffer.get(i).yaw;

                }

                try {
                    //tflite
                    model.run(input, output);
                    steering= output[0][0];
                    speed = output[0][1];
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }



    private class buf{
        float pitch,roll,yaw;
        float x,y,z;
        float speed, steering;

        private buf(float pitch, float roll, float yaw, float x, float y, float z, float speed, float steering){
            this.pitch = pitch;
            this.roll = roll;
            this.yaw = yaw;
            this.x = x;
            this.y = y;
            this.z = z;
            this.speed = speed;
            this.steering = steering;
        }
    }

    private class GyroscopeListener implements SensorEventListener {

        @Override
        public void onSensorChanged(SensorEvent event) {

            /* 각 축의 각속도 성분을 받는다. */
            double gyroX = event.values[0];
            double gyroY = event.values[1];
            double gyroZ = event.values[2];

            /* 각속도를 적분하여 회전각을 추출하기 위해 적분 간격(dt)을 구한다.
             * dt : 센서가 현재 상태를 감지하는 시간 간격
             * NS2S : nano second -> second */
            dt = (event.timestamp - timestamp) * NS2S;
            timestamp = event.timestamp;

            /* 맨 센서 인식을 활성화 하여 처음 timestamp가 0일때는 dt값이 올바르지 않으므로 넘어간다. */
            if (dt - timestamp*NS2S != 0) {

                /* 각속도 성분을 적분 -> 회전각(pitch, roll)으로 변환.
                 * 여기까지의 pitch, roll의 단위는 '라디안'이다.
                 * SO 아래 로그 출력부분에서 멤버변수 'RAD2DGR'를 곱해주어 degree로 변환해줌.  */
                pitch = pitch + gyroY*dt;
                roll = roll + gyroX*dt;
                yaw = yaw + gyroZ*dt;

                if(pitch >= 2*Math.PI)    pitch -= 2*Math.PI;
                if(roll >= 2*Math.PI)    roll -= 2*Math.PI;
                if(yaw >= 2*Math.PI)    yaw -= 2*Math.PI;
                if(pitch <= -2*Math.PI)    pitch += 2*Math.PI;
                if(roll <= -2*Math.PI)    roll += 2*Math.PI;
                if(yaw <= -2*Math.PI)    yaw += 2*Math.PI;


            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    }

    private class AccelerometerListener implements SensorEventListener {

        @Override
        public void onSensorChanged(SensorEvent event) {

            X = event.values[0];
            Y = event.values[1];
            Z = event.values[2];

        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    }

    @Override
    public Context getBaseContext() {
        return super.getBaseContext();
    }

    @Override
    public void onDestroy(){
        super.onDestroy();

        this.finish();
    }

}
