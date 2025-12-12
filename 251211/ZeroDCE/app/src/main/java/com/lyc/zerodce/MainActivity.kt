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

    // UI状态
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var enhancedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    // 初始化模型
    val enhancer = remember { ZeroDCEEnhancer(context) }

    // 系统照片选择器
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        uri?.let { selectedUri ->
            scope.launch {
                // 1. 读取图片
                val bitmap = loadBitmapFromUri(context, selectedUri)
                originalBitmap = bitmap
                enhancedBitmap = null // 清空旧结果

                // 2. 开始增强
                if (bitmap != null) {
                    isProcessing = true
                    withContext(Dispatchers.Default) {
                        val result = enhancer.enhance(bitmap)
                        withContext(Dispatchers.Main) {
                            enhancedBitmap = result
                            isProcessing = false
                        }
                    }
                }
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Zero-DCE 微光增强",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // === 增强后结果 ===
        if (enhancedBitmap != null) {
            Text("增强后 (256x256)", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
            Card(
                modifier = Modifier.padding(8.dp).size(256.dp),
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Image(
                    bitmap = enhancedBitmap!!.asImageBitmap(),
                    contentDescription = "Enhanced",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds // 拉伸填满
                )
            }
            Button(
                onClick = { saveBitmapToGallery(context, enhancedBitmap!!) },
                modifier = Modifier.padding(vertical = 8.dp)
            ) {
                Text("保存到相册")
            }
            Divider(modifier = Modifier.padding(vertical = 16.dp))
        }

        // === 原图预览 ===
        if (originalBitmap != null) {
            Text("原图", style = MaterialTheme.typography.titleSmall)
            Card(modifier = Modifier.padding(8.dp).height(200.dp)) {
                Image(
                    bitmap = originalBitmap!!.asImageBitmap(),
                    contentDescription = "Original",
                    contentScale = ContentScale.Fit
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // === 按钮区 ===
        if (isProcessing) {
            CircularProgressIndicator()
            Text("正在处理中...", modifier = Modifier.padding(top = 8.dp))
        } else {
            Button(
                onClick = {
                    // 打开只看图片的选择器
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                modifier = Modifier.fillMaxWidth().height(50.dp)
            ) {
                Text(if (originalBitmap == null) "从相册选择图片" else "换一张图片")
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