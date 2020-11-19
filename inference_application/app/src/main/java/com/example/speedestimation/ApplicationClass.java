package com.example.speedestimation;

// ApplicationClass.java

import android.app.Application;
import android.content.Context;

public class ApplicationClass extends Application {
    private static android.content.res.Resources mContext;
    private static Context x;

    public static void setcnxt(Context c){
        x = c;
    }
    public static Context getcnxt(){
        return x;
    }

    public static void setContext(android.content.res.Resources c){
        mContext = c;
    }

    public static android.content.res.Resources getContext() {
        return mContext;
    }
}
