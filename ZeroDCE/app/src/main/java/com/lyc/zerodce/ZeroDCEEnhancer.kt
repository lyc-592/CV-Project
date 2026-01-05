package com.lyc.zerodce

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ZeroDCEEnhancer(context: Context) {
    private var interpreter: Interpreter? = null
    private val modelName = "ZeroDCE_dynamic_float32.tflite"

    // === 修改 1: 提高默认画质上限 ===
    // 2048px 对于中高端手机是可接受的，能保留绝大部分细节
    // 如果想要极致清晰且不怕慢，可以尝试 2560 或 3000
    private val highQualitySize = 2048
    private val lowQualitySize = 1024

    init {
        val options = Interpreter.Options()
        // 强烈建议开启 GPU，否则跑大图会非常慢
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

    /**
     * 对外暴露的增强接口
     * 自动处理分辨率回退逻辑
     */
    fun enhance(originalBitmap: Bitmap): Bitmap? {
        interpreter ?: return null

        val originalW = originalBitmap.width
        val originalH = originalBitmap.height

        // 尝试使用高质量尺寸运行
        var result = runInferenceSafe(originalBitmap, highQualitySize)

        // 如果高质量失败（通常是 OOM），则降级使用低质量尺寸
        if (result == null) {
            Log.w("ZeroDCE", "高质量推理失败 (OOM)，尝试降级处理...")
            result = runInferenceSafe(originalBitmap, lowQualitySize)
        }

        // 如果还失败，返回空
        if (result == null) return null

        // === 修改 2: 强制还原回原图尺寸 ===
        // 这一步至关重要！虽然 result 可能是缩小的，
        // 但我们要用双线性插值（filter=true）把它拉伸回原图大小。
        // 这样用户在相册里看到的图才不会变小。
        if (result.width != originalW || result.height != originalH) {
            Log.d("ZeroDCE", "正在还原尺寸: ${result.width}x${result.height} -> ${originalW}x${originalH}")
            val finalBitmap = Bitmap.createScaledBitmap(result, originalW, originalH, true)
            // 释放中间的小图内存
            if (result != finalBitmap) {
                result.recycle()
            }
            return finalBitmap
        }

        return result
    }

    /**
     * 内部安全的推理函数，包含 OOM 捕获
     */
    private fun runInferenceSafe(originalBitmap: Bitmap, maxSize: Int): Bitmap? {
        try {
            // 1. 计算缩放尺寸
            val width = originalBitmap.width
            val height = originalBitmap.height
            var newWidth = width
            var newHeight = height

            if (width > maxSize || height > maxSize) {
                val ratio = width.toFloat() / height.toFloat()
                if (width > height) {
                    newWidth = maxSize
                    newHeight = (maxSize / ratio).toInt()
                } else {
                    newHeight = maxSize
                    newWidth = (maxSize * ratio).toInt()
                }
            }

            // 确保尺寸是 4 的倍数 (有些 GPU Delegate 对非对齐尺寸敏感)
            newWidth = (newWidth / 4) * 4
            newHeight = (newHeight / 4) * 4

            if (newWidth <= 0 || newHeight <= 0) return null

            Log.d("ZeroDCE", "尝试推理尺寸: $newWidth x $newHeight")

            // 缩放输入图
            val scaledBitmap = Bitmap.createScaledBitmap(originalBitmap, newWidth, newHeight, true)

            // 2. 动态调整 Input Tensor
            interpreter?.resizeInput(0, intArrayOf(1, newHeight, newWidth, 3))
            interpreter?.allocateTensors() // 这一步最容易 OOM

            // 3. 准备输入数据
            val inputBuffer = ByteBuffer.allocateDirect(1 * newHeight * newWidth * 3 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            val intValues = IntArray(newWidth * newHeight)
            scaledBitmap.getPixels(intValues, 0, newWidth, 0, 0, newWidth, newHeight)

            // 及时回收 scaledBitmap 节省内存
            if (scaledBitmap != originalBitmap) {
                scaledBitmap.recycle()
            }

            // 填充数据 (0-255 -> 0.0-1.0)
            for (pixel in intValues) {
                inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel and 0xFF) / 255.0f))
            }

            // 4. 准备输出数据
            val outputBuffer = ByteBuffer.allocateDirect(1 * newHeight * newWidth * 3 * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // 5. 运行
            interpreter?.run(inputBuffer, outputBuffer)

            // 6. 转回 Bitmap
            outputBuffer.rewind()
            val outputPixels = IntArray(newWidth * newHeight)
            val floatBuffer = outputBuffer.asFloatBuffer()

            for (i in 0 until newWidth * newHeight) {
                val r = (floatBuffer.get() * 255.0f).toInt().coerceIn(0, 255)
                val g = (floatBuffer.get() * 255.0f).toInt().coerceIn(0, 255)
                val b = (floatBuffer.get() * 255.0f).toInt().coerceIn(0, 255)
                outputPixels[i] = Color.rgb(r, g, b)
            }

            return Bitmap.createBitmap(outputPixels, newWidth, newHeight, Bitmap.Config.ARGB_8888)

        } catch (e: OutOfMemoryError) {
            Log.e("ZeroDCE", "内存溢出: ${e.message}")
            return null
        } catch (e: Exception) {
            Log.e("ZeroDCE", "推理错误: ${e.message}")
            return null
        }
    }

    fun close() {
        interpreter?.close()
    }
}