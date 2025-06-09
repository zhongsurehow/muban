

# 📝《逐帧动画导出模板（标准通用版）使用说明》

---

## 🎯 模板目的

本模板用于**快速制作逐帧动画，并导出为 PNG 图片 ZIP 包**，可用于：

✅ 短视频配图
✅ 带动效的幻灯片 / 视频背景
✅ AI生成视频素材（与其他视频剪辑软件配合）
✅ 任意帧控动画

---

## 🗂️ 模板结构

模板分为 **5大区块**，只改指定区块，核心逻辑区**禁止修改**，保证模板长期复用稳定。

### 1️⃣ 【基础配置区】👉 需要改

配置画布大小、帧率、动画总时长。

```js
const CANVAS_WIDTH = 1920;         // 画布宽度
const CANVAS_HEIGHT = 1080;        // 画布高度
const FRAME_RATE = 30;             // 帧率
const TOTAL_DURATION = 5;          // 动画总时长（秒）
const TOTAL_FRAMES = Math.ceil(TOTAL_DURATION * FRAME_RATE);
```

**注意事项：**

* 一般画布固定为 `1920x1080`（高清视频标准）。
* `FRAME_RATE` 推荐 `30`，需要低帧可改 `15`。
* 每次动画改 `TOTAL_DURATION`，影响总帧数。

---

### 2️⃣ 【动画元素定义区】👉 需要改

定义本动画要用到的 **变量/对象**，例如：

```js
const textObj = {
    x: CANVAS_WIDTH / 2,
    y: CANVAS_HEIGHT / 2,
    fontSize: 100,
    text: '🚀 启动！',
    opacity: 1
};
```

**你可以定义多个元素**，如：

```js
const textObj1 = { ... }
const textObj2 = { ... }
const rectObj = { ... }
```

---

### 3️⃣ 【动画 timeline 定义区】👉 需要改

定义 **动画变化过程**，使用 `GSAP timeline`：

```js
const timeline = gsap.timeline({ paused: true })
    .to(textObj, { x: 1600, duration: 3, ease: 'power2.inOut' })
    .to(textObj, { fontSize: 200, duration: 2, ease: 'elastic.out' }, 1)
    .to(textObj, { opacity: 0, duration: 1 }, 4);
```

**注意事项：**

* `timeline` 必须 paused: true，模板负责手动推进 progress。
* **所有动画逻辑都写在这里**，AI不要改 `exportFrame` 中的逻辑。

---

### 4️⃣ 【drawScene 函数】👉 每次动画自定义画面内容

**核心自定义绘制逻辑**，每一帧调用 `drawScene`，你在这里定义如何绘制当前帧：

```js
const drawScene = () => {
    ctx.globalAlpha = textObj.opacity;
    ctx.font = `${textObj.fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#000';
    ctx.fillText(textObj.text, textObj.x, textObj.y);
    ctx.globalAlpha = 1;
};
```

✅ **可以绘制多个元素**，支持：

* 文字 / emoji
* 矩形、圆形、线条
* 图片 (需要用 `drawImage`)
* 背景渐变 / 背景图片

**模板要求：**
**不要改 `drawFrame` 函数，只改 `drawScene`。**
**AI理解重点：** drawScene 定义 "我每一帧画什么内容"，越独立越好，便于复用。

---

### 5️⃣ 【核心逻辑区】👉 🚫 不要改 🚫

```js
const drawFrame = () => { ... };
const exportFrame = () => { ... };
按钮绑定逻辑 ...
```

**此区块为固定模板**：

✅ 保证帧数一致推进
✅ 负责调用 timeline.progress()
✅ 调用 drawFrame → drawScene
✅ 导出 ZIP 包逻辑稳定可靠

👉 **AI 请不要修改这部分逻辑，否则可能导致导出异常 / 帧错位。**
👉 AI 生成新动画只需改 1️⃣ 2️⃣ 3️⃣ 4️⃣ 区，不需要改 5️⃣ 核心逻辑区。

---

## 🚀 使用流程

1️⃣ 配置【基础配置区】 → 改时长 / 帧率
2️⃣ 定义【动画元素定义区】 → 定义本次要用的变量
3️⃣ 配置【动画 timeline 定义区】 → 定义动画变化
4️⃣ 修改【drawScene】 → 定义如何绘制每帧
5️⃣ 点击 "开始导出" → 导出 PNG 图片 → ZIP 包下载

---

## ❗ 特别注意

✅ `drawScene` 应只做绘制逻辑，**不要更新动画数据**（动画数据由 GSAP timeline 控制）。
✅ `drawFrame`、`exportFrame`、按钮逻辑**不要改**，模板保证稳定性。
✅ 导出 PNG 时背景默认白色，若需要透明 PNG，可将：

```js
ctx.fillStyle = '#fff';
ctx.fillRect(...);
```

改为：

```js
ctx.clearRect(0, 0, canvas.width, canvas.height);
```

即可支持透明背景。

---

## ✅ AI 使用重点提示

* AI 在复用模板时，只修改：

  * 基础配置区
  * 动画元素定义区
  * 动画 timeline 定义区
  * drawScene 函数
* 核心逻辑区不可改。
* AI 不要自动改 `exportFrame` / `drawFrame` 函数内部逻辑。

