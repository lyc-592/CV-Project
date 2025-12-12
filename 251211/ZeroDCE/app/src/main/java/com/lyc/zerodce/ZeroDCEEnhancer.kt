package com.lyc.zerodce

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ZeroDCEEnhancer(context: Context) {
    private var interpreter: Interpreter? = null
    // 确保这里的名字和你assets里的文件名一模一样
    private val modelName = "ZeroDCE_float32.tflite"
    private val inputSize = 256

    init {
        val options = Interpreter.Options()
        // 尝试开启GPU加速
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            options.addDelegate(GpuDelegate())
        } else {
            options.setNumThreads(4)
        }

        try {
            val assetFileDescriptor = context.assets.openFd(modelName)
            val fileInputStream = java.io.FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            interpreter = Interpreter(mappedByteBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun enhance(originalBitmap: Bitmap): Bitmap? {
        interpreter ?: return null

        // 1. 强制缩放到 256x256 (模型要求)
        val scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, inputSize, inputSize, true)

        // 2. 准备输入数据 [1, 256, 256, 3]
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        scaledBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        // 像素归一化: 0-255 -> 0.0-1.0
        for (pixel in intValues) {
            inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
            inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
            inputBuffer.putFloat(((pixel and 0xFF) / 255.0f))
        }

        // 3. 准备输出数据
        val outputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // 4. 运行模型
        interpreter?.run(inputBuffer, outputBuffer)

        // 5. 数据转回 Bitmap
        outputBuffer.rewind()
        val outputPixels = IntArray(inputSize * inputSize)
        for (i in 0 until inputSize * inputSize) {
            val r = (outputBuffer.float * 255.0f).toInt().coerceIn(0, 255)
            val g = (outputBuffer.float * 255.0f).toInt().coerceIn(0, 255)
            val b = (outputBuffer.float * 255.0f).toInt().coerceIn(0, 255)
            outputPixels[i] = Color.rgb(r, g, b)
        }

        return Bitmap.createBitmap(outputPixels, inputSize, inputSize, Bitmap.Config.ARGB_8888)
    }
}