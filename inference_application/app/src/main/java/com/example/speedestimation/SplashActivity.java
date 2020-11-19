package com.example.speedestimation;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class SplashActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        final Context content = this;

        // tflite 모델 파일 존재 여부 체크
        TextView splash_text = (TextView)findViewById(R.id.splash_text);
        splash_text.setText("tflite file check...");
        // 센서 동작 여부(acc, gyro) 체크
        splash_text.setText("sensor check...");
        // 완료
        splash_text.setText("Touch to Start");

        try {
            Thread.sleep(1000);
        }catch (InterruptedException e){
            e.printStackTrace();
        }
        splash_text.setOnClickListener(new TextView.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent MainActivityIntent = new Intent(SplashActivity.this, MainActivity.class);
                startActivity(MainActivityIntent);
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();

        // Remove the activity when its off the screen
        finish();
    }
}
