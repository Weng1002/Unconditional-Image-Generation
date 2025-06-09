# Unconditional-Image-Generation
113-2 電機所 生成式AI HW5 Unconditional Image Generation

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第五次作業，實作一個 Unconditional Image Generation（無條件影像生成） 模型，並以 PTT 表特板上的人臉圖片作為訓練資料，學習其潛在分佈特徵。

將需從PTT 表特板文章中蒐集人臉圖片，經過適當的前處理後，利用生成模型學習資料分佈，並產出 64x64 解析度的人臉圖片。

最終所生成的圖片將與測試資料進行分佈比較，透過 Fréchet Inception Distance（FID）作為評估指標，測量生成影像與真實資料的相似程度。

[連結](https://nycubasic.duckdns.org/competitions/4/#/pages-tab) 

## 任務&目標
