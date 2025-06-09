# Unconditional-Image-Generation
113-2 電機所 生成式AI HW5 Unconditional Image Generation

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第五次作業，實作一個 Unconditional Image Generation（無條件影像生成） 模型，並以 PTT 表特板上的人臉圖片作為訓練資料，學習其潛在分佈特徵。

將需從PTT 表特板文章中蒐集人臉圖片，經過適當的前處理後，利用生成模型學習資料分佈，並產出 64x64 解析度的人臉圖片。

最終所生成的圖片將與測試資料進行分佈比較，透過 Fréchet Inception Distance（FID）作為評估指標，測量生成影像與真實資料的相似程度。

[任務連結](https://nycubasic.duckdns.org/competitions/4/#/pages-tab) 

## 操作流程

### 1. Crawl

(1) 先爬取特定年份的所有 PTT 表特板 上的文章

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
