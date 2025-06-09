# Unconditional-Image-Generation
113-2 電機所 生成式AI HW5 Unconditional Image Generation

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第五次作業，實作一個 Unconditional Image Generation（無條件影像生成） 模型，並以 PTT 表特板上的人臉圖片作為訓練資料，學習其潛在分佈特徵。

將需從PTT 表特板文章中蒐集人臉圖片，經過適當的前處理後，利用生成模型學習資料分佈，並產出 64x64 解析度的人臉圖片。

最終所生成的圖片將與測試資料進行分佈比較，透過 Fréchet Inception Distance（FID）作為評估指標，測量生成影像與真實資料的相似程度。

[任務連結](https://nycubasic.duckdns.org/competitions/4/#/pages-tab) 

## 操作流程

### 1. Crawl

(1) 先爬取特定年份的所有 PTT 表特板 上的文章，我這邊使用動態調整PTT 的列表ID，來控制年份在 2020-2025。

```
  def extract_meta_value(soup, label):
    tags = soup.select('span.article-meta-tag')
    vals = soup.select('span.article-meta-value')

    for tag, val in zip(tags, vals):
        if tag.text.strip() == label:
            return val.text.strip()
    
    # 處理特殊情況，沒有時間欄位時
    if label == '時間':
        f2_texts = [span.text.strip() for span in soup.select('span.f2')]

        for line in f2_texts:
            match = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})', line)
            if match:
                date_part = match.group(1)  
                time_part = match.group(2)  
                dt = datetime.strptime(date_part + ' ' + time_part, '%m/%d/%Y %H:%M:%S')
                return dt.strftime('%a %b %d %H:%M:%S %Y')

    return None


def crawl_articles():
    is_2024_started = False
    start_index = 2329  #3911(2025)  3647(2024)   3367(2023)    2712(2021)  2329(2020)
    end_index = 2712   #4001(2025)  3916(2024)   3647(2023)   3060(2021)   2712(2020)

    for index in range(start_index, end_index+1):  
        url = f'https://www.ptt.cc/bbs/Beauty/index{index}.html'
        print("\n")
        print(f'目前的列表: {url}')

        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status() # 檢查是否取得成功
        soup = BeautifulSoup(res.text, 'html.parser')

        entries = soup.select('div.r-ent') # 取得文章列表
        for entry in entries:
            link_tag = entry.select_one('a')
            if not link_tag:
                continue  # 無網址的文章

            post_url = 'https://www.ptt.cc' + link_tag['href']
            title_text = link_tag.text.strip()
            date_tag = entry.select_one('div.date')
            post_date = date_tag.text.strip() if date_tag else ''
            print(f'抓到的文章：{post_url} | 標題：{title_text} | 列表時間：{post_date}')
            

            # 還沒進入 2024，先用舊邏輯
            if not is_2024_started:
                res_post = requests.get(post_url, headers=HEADERS, timeout=10)
                res_post.raise_for_status()
                post_soup = BeautifulSoup(res_post.text, 'html.parser')
                post_time = extract_meta_value(post_soup, '時間')
                if not post_time:
                    print(f"無法解析時間")
                    continue
                dt = datetime.strptime(post_time, '%a %b %d %H:%M:%S %Y')

                if dt.year < 2020:
                    print('跳過早於 2024 年的文章')
                    continue
                elif dt.year > 2020:
                    print('跳過 2025 年的文章')
                    continue

                # 確認已進入 2024 年
                is_2024_started = True
                mmdd = dt.strftime('%m%d')
            else:
                # 已進入 2024，直接從列表時間推斷
                if post_date >= '01/01':
                    if res_post.status_code == 404:
                        print("該文章為 404，略過")
                        continue
                    post_soup = BeautifulSoup(res_post.text, 'html.parser')
                    post_time = extract_meta_value(post_soup, '時間')
                    if not post_time:
                        continue
                    dt = datetime.strptime(post_time, '%a %b %d %H:%M:%S %Y')

                    if dt.year == 2021:
                        print("抓到 2025 年文章，結束爬蟲")
                        return
                    elif dt.year != 2020:
                        continue        
                    mmdd = dt.strftime('%m%d')
                else:
                    mmdd = post_date.replace("/", "").zfill(4)

            # 篩選
            if not title_text.strip():
                print(f'略過標題為空白或空字串')
                continue
            if '[公告]' in title_text or 'Fw:[公告]' in title_text:
                print('略過公告文')
                continue
            
            article_data = {
                'date': mmdd,
                'title': title_text,
                'url': post_url
            }
            
            with open('2020_articles.jsonl', 'a', encoding='utf-8') as fa:
                fa.write(json.dumps(article_data, ensure_ascii=False) + '\n')

            
            time.sleep(random.uniform(0.3, 0.5)) 
```

(2) 接著只截取包含 **正妹** 關鍵字的文章 ( 包含內文 )

```
  def extract_image_urls(text):
    pattern = r'https?://[^\s"]+\.(?:jpg|jpeg|png|gif)(?=\b|$)'
    return re.findall(pattern, text, flags=re.IGNORECASE)

def Keyword(keyword: str):
    target_articles = []
    image_urls = []

    with open('2020_articles.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)

            if keyword in article["title"]: # 標題包含關鍵字
                target_articles.append(article)
                continue

            # 否則檢查內文是否包含關鍵字
            url = article["url"]
            res = requests.get(url, headers=HEADERS, timeout=40)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            main_content = soup.select_one("#main-content")
            if not main_content:
                continue

            text = main_content.get_text(separator="\n")
            content_split = text.split("※ 發信站")
            if len(content_split) < 2:
                continue
            content = content_split[0]

            if keyword in content:
                target_articles.append(article)

    print(f"找到 {len(target_articles)} 篇文章（標題或內文含關鍵字「{keyword}」）")

    for article in tqdm(target_articles, desc="處理特定文章"):
        url = article["url"]
        print(f"處理中：{url}")
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
        except Exception as e:
            print(f"[!] 無法下載文章內容：{e}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        main_content = soup.select_one("#main-content")
        if not main_content:
            continue

        text = main_content.get_text(separator="\n")
        content_split = text.split("※ 發信站")
        if len(content_split) < 2:
            print("無法找到發信站標記，跳過")
            continue

        content = content_split[0]

        print("符合條件，開始擷取圖片連結")

        pushes = soup.select("div.push span.push-content")
        for push in pushes:
            content += push.text

        image_urls += extract_image_urls(content)
        time.sleep(random.uniform(0.1, 0.3))

    unique_image_urls = list(set(image_urls))

    result = {
        "image_urls": unique_image_urls
    }

    outname = f"2020_keyword_{keyword}.json"
    with open(outname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"完成 keyword 抽圖：{outname}，共 {len(unique_image_urls)} 張圖片")

```

(3) 將擁有 **正妹** 中的 img_urls 都下載到 raw_images ( 因為我伺服器的硬碟不夠大，所以我後面處理 raw_images 後就刪除原照片 )

```
  save_dir = '../raw_images/2020_正妹_images'
os.makedirs(save_dir, exist_ok=True)

with open('2020_keyword_正妹.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    image_urls = data.get("image_urls", [])

print(f"共載入 {len(image_urls)} 張圖片網址，開始下載")

# 開始下載圖片
for idx, url in enumerate(tqdm(image_urls, desc="Downloading")):
    if url.startswith("https://d.img.vision/dddshay/"):
        print(f"[!] 跳過：{url}")
        continue
    try:
        response = requests.get(
            url,
            timeout=35,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'Referer': 'https://www.google.com/'
            }
        )
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        filename = f"{idx:05d}.png" # 儲存為 PNG 格式
        filepath = os.path.join(save_dir, filename)
        image.save(filepath, format="PNG")

    except Exception as e:
        print(f"[!] 第 {idx} 張圖片處理失敗：{url} | 錯誤：{e}")
```

### 2. Filter_face

(1) 利用 OpenCV 去篩選出只有正臉的人臉照

(2) 設定一些人臉辨識的條件。

#### 雙眼亮度判斷（濾掉遮眼、瀏海、墨鏡）

```
  def eye_region_brightness(img, left_eye, right_eye):
      patch_size = 10  # pixel
      eyes = [left_eye, right_eye]
      brightness = []
      for eye in eyes:
          x, y = int(eye[0]), int(eye[1])
          patch = img[max(0, y - patch_size):y + patch_size, max(0, x - patch_size):x + patch_size]
          if patch.size == 0:
              return 0  # 若 patch 無效則視為太暗
          gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
          brightness.append(np.mean(gray))
      return np.mean(brightness)
```

#### 雙眼距離過短 → 側臉（或單眼）

```
  left_eye, right_eye = face.kps[0], face.kps[1]
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 15:
            continue
```

#### 雙眼亮度過暗 → 遮眼（瀏海、墨鏡、閉眼）

```
  brightness = eye_region_brightness(img, left_eye, right_eye)
          if brightness < 60:
              continue
```

#### 模糊判斷（Laplacian）

```
  if is_blurry(face_crop, threshold=50):
            continue
```




