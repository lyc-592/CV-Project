package com.lyc.zerodce

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.provider.MediaStore
import android.widget.Toast
import java.io.OutputStream

// 保存图片到相册的函数
fun saveBitmapToGallery(context: Context, bitmap: Bitmap) {
    val filename = "ZeroDCE_${System.currentTimeMillis()}.jpg"
    var fos: OutputStream? = null
    var imageUri: android.net.Uri? = null

    try {
        val resolver = context.contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/ZeroDCE")
        }

        imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        fos = imageUri?.let { resolver.openOutputStream(it) }

        fos?.let {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
            Toast.makeText(context, "保存成功！请去相册查看", Toast.LENGTH_LONG).show()
        }
    } catch (e: Exception) {
        Toast.makeText(context, "保存失败: ${e.message}", Toast.LENGTH_SHORT).show()
    } finally {
        fos?.close()
    }
}