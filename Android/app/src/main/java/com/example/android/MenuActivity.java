package com.example.android;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.content.Intent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

public class MenuActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_menu);

        // 获取按钮
        ImageButton DiseaseDetectButton = findViewById(R.id.health);

        // 设置点击事件监听器
        DiseaseDetectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 创建一个意图，用于启动 MenuActivity
                Intent intent = new Intent(MenuActivity.this, DiseaseDetectActivity.class);
                startActivity(intent);
            }
        });
    }
}