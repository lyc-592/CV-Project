package com.lyc.zerodce

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.lyc.zerodce.ui.theme.ZeroDCETheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ZeroDCETheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    // UI状态
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var enhancedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    // 初始化模型 (使用 remember 保持实例)
    val enhancer = remember { ZeroDCEEnhancer(context) }

    // 当 Composable 销毁时，关闭模型释放资源
    DisposableEffect(Unit) {
        onDispose {
            enhancer.close()
        }
    }

    // 系统照片选择器
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        uri?.let { selectedUri ->
            scope.launch {
                isProcessing = true
                // 1. 读取图片
                val bitmap = loadBitmapFromUri(context, selectedUri)

                if (bitmap != null) {
                    originalBitmap = bitmap
                    enhancedBitmap = null // 清空旧结果

                    // 2. 开始增强 (在后台线程运行)
                    withContext(Dispatchers.Default) {
                        val result = enhancer.enhance(bitmap)
                        // 切回主线程更新 UI
                        withContext(Dispatchers.Main) {
                            enhancedBitmap = result
                            isProcessing = false
                        }
                    }
                } else {
                    isProcessing = false
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(scrollState), // 允许内容滚动
        horizontalAlignment = Alignment.CenterHorizontally,
        // 【核心修改】这里改为 Center，让内容在屏幕中间
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Zero-DCE 微光增强",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // === 增强后结果展示区 ===
        if (enhancedBitmap != null) {
            Text(
                text = "增强后 (${enhancedBitmap!!.width}x${enhancedBitmap!!.height})",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight(), // 高度随图片比例自适应
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Image(
                    bitmap = enhancedBitmap!!.asImageBitmap(),
                    contentDescription = "Enhanced Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 450.dp), // 限制最大高度，防止长图占满全屏
                    contentScale = ContentScale.Fit // 关键：保持比例，不拉伸
                )
            }

            Button(
                onClick = { saveBitmapToGallery(context, enhancedBitmap!!) },
                modifier = Modifier
                    .padding(vertical = 16.dp)
                    .fillMaxWidth()
            ) {
                Text("保存到相册")
            }

            Divider(modifier = Modifier.padding(vertical = 16.dp))
        }

        // === 原图预览区 ===
        if (originalBitmap != null) {
            Text(
                text = "原图 (${originalBitmap!!.width}x${originalBitmap!!.height})",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Image(
                    bitmap = originalBitmap!!.asImageBitmap(),
                    contentDescription = "Original Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 250.dp), // 原图预览稍微小一点
                    contentScale = ContentScale.Fit
                )
            }

            Spacer(modifier = Modifier.height(24.dp))
        }

        // === 底部操作按钮 ===
        if (isProcessing) {
            CircularProgressIndicator()
            Text("正在增强中...", modifier = Modifier.padding(top = 16.dp))
        } else {
            Button(
                onClick = {
                    // 打开只看图片的选择器
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Text(
                    text = if (originalBitmap == null) "从相册选择图片" else "换一张图片",
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }
    }
}

// 辅助函数：Uri转Bitmap
suspend fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return withContext(Dispatchers.IO) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(context.contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}