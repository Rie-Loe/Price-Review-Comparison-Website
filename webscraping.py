# Programmer Name: Loe Hui Lin
# TP Number      : TP060359
# Intake Code    : APD3F2205CS(DA)
# Course         : BSc(Hons) Computer Science in Data Analytics

# Import libraries
import requests # Web scraping
from bs4 import BeautifulSoup
import json
import nltk # Data preprocessing
try:
    nltk.data.find('C:/Users/Loe Hui Lin/AppData/Roaming/nltk_data/corpora/wordnet')
    print("nltk wordnet found")
except LookupError:
    nltk.download('wordnet')
    print("ahhhhh I have to download wordnet again")
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
import pandas as pd # Data manipulation
pd.options.mode.chained_assignment = None  # default='warn'   # to drop rows without showing warning
import re
import csv
from fuzzywuzzy import fuzz # String matching
import matplotlib.pyplot as plt # Visualization
import seaborn as sns
from wordcloud import WordCloud
import pickle # Saving model 

from typing import List, Set, Dict, Tuple

def run_application(searched_item: str):
    # Web scrape
    lazada_searches, shopee_searches = web_scrape_product_info (searched_item)
    print("web scraping: passed")

    # Filter returned searches
    lazada_searches = filter_search(lazada_searches, searched_item)
    shopee_searches = filter_search(shopee_searches, searched_item)
    print("filter searches: passed")

    # Scrape reviews
    laz_review_list = []
    for row in lazada_searches.itertuples():
        laz_product_rating = get_review_lazada(str(row[7])) 
        laz_review_list.append(laz_product_rating)
    lazada_searches['scraped_review'] = laz_review_list

    sho_review_list = []
    for row in shopee_searches.itertuples():
        sho_product_rating = get_review_shopee(str(row[8]), str(row[7])) 
        sho_review_list.append(sho_product_rating)
    shopee_searches['scraped_review'] = sho_review_list
    print("scrape reviews: passed")

    # Predict sentiments
    final_lazada_output, final_lazada_df = summarised_prediction(laz_review_list, lazada_searches)
    final_shopee_output, final_shopee_df = summarised_prediction(sho_review_list, shopee_searches)
    print("predict sentiments: passed")

    # Create long dataframes
    lazada_long_product = review_long_df(final_lazada_output, final_lazada_df)
    shopee_long_product = review_long_df(final_shopee_output, final_shopee_df)
    lazada_long_product.to_csv("lazada_long_product.csv", index=False, header=True)
    shopee_long_product.to_csv("shopee_long_product.csv", index=False, header=True)
    print("create long dfs: passed")

    # Remove Invalid Reviews - Removing before long dataframe will cause errors
    final_lazada_df = remove_invalid_reviews(final_lazada_df)
    final_shopee_df = remove_invalid_reviews(final_shopee_df)
    final_lazada_df.to_csv("lazada_product_df.csv", index=False, header=True)
    final_shopee_df.to_csv("shopee_product_df.csv", index=False, header=True)
    print("remove invalid reviews: passed")

    # Create final platform dataframes
    final_df = merge_dataframes(final_lazada_df, final_shopee_df)
    final_df.to_csv("final_df.csv", index=False, header=True)
    print("create final dfs: passed")

    # Create long reviews
    lazada_pos_review, lazada_neg_review = wordcloud_review_df(lazada_long_product)
    lazada_pos_review.to_csv("wordcloud_lazada_pos_review.csv", index=False, header=True)
    lazada_neg_review.to_csv("wordcloud_lazada_neg_review.csv", index=False, header=True)
    shopee_pos_review, shopee_neg_review = wordcloud_review_df(shopee_long_product)
    shopee_pos_review.to_csv("wordcloud_shopee_pos_review.csv", index=False, header=True)
    shopee_neg_review.to_csv("wordcloud_shopee_neg_review.csv", index=False, header=True)
    print("create long reviews: passed")

    # Run visualisations
    
    return


def scrape_lazada_info(searched_item: str) -> pd.DataFrame:
    """
        Scrape product information in first page of Lazada
        
        Parameters:
        searched_item: Item to request for
        
        Returns:
        A dataframe of search results for related products
    """
    # Create lists to store different product information
    lazada_product_title_list = list()
    lazada_product_price_min_list = list()
    lazada_product_price_max_list = list()
    lazada_product_rating_list = list()
    lazada_product_star_list = list()
    lazada_product_sold_list = list()
    lazada_product_itemid_list = list()
    lazada_product_shopid_list = list()
    lazada_product_itemurl_list = list()

    # Manual method of obtaining Request URL: F12 -> Network -> Fetch/XHR
    # Automate manual method by replacing keyword with searched_item
    lazada_url = 'https://www.lazada.sg/catalog/?_keyori=ss&ajax=true&from=input&isFirstRequest=true&page=1&q=resin%20tea%20coaster&spm=a2o42.home.search.go.654346b5Nj9HsG'
    if ' ' in searched_item:
        lazada_searched_item = searched_item.replace(" ", "%20")
        new_lazada_url = lazada_url.replace('resin%20tea%20coaster', lazada_searched_item)
    else:
        new_lazada_url = lazada_url.replace('resin%20tea%20coaster', searched_item)

    header = {
        "authority": "member.lazada.sg",
        "method": "GET",
        "path": "/user/api/getUser",
        "scheme": "https",
        "accept": "application/json, text/javascript",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9,ms;q=0.8",
        "cookie": "client_type=desktop; lzd_cid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; t_uid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; lzd_sid=115065e370eb173c150b89b82f3657c2; anon_uid=e0abc68e7864a7414fce4c16e60580b4; _tbtoken=eee3b383373e0; t_fv=1668525165789; _gcl_au=1.1.1770287619.1668525166; hng=SG|en-SG|SGD|702; hng.sig=ryBKXOqZIsp9xOQ3YsZRgD7f-p0UaGB2pZ4BbZM8uEc; xlly_s=1; _m_h5_tk=7b313bd7db4ef975bc78b67fc7d4643e_1672938834121; _m_h5_tk_enc=a41920d8e3af937201c0f0245c75f9eb; t_sid=0SuQOSPKsOv0mJcSGimyZBJO0cvmrGzm; utm_channel=NA; x5sec=7b22617365727665722d6c617a6164613b32223a223836376635353730373131383033663036333263643633666662633339313131434d373932353047454d66452f5a693367706a454444434b6d5a6e332b2f2f2f2f2f384251414d3d227d; isg=BHZ2n7xICXfCuP1WvJP7BSCIx6x4l7rRJmHRO-BcINn0Ixe9SyWu4BuaO_dPi7Lp; l=fBOgmYC7T1fHvQtZBO5Courza779EQOb8sPzaNbMiIEGa6NG9FZdHNCFz5UeWdtj_T55-e-zEHwxcdeMWmaU-jkDBeYQ-JDocaJw-ewZ_o7d.; tfstk=cuNdBr1l-GjnM1So_XBMaYJ70rQGaUj-feikwSOwxPjMaGdkasbuiS3Xj9gS0DQO.",
        "dnt": "1",
        "sec-ch-ua": '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    }

    # Make request to website
    response = requests.get(new_lazada_url, headers=header)
    response_text = response.text

    # Convert json to Python dictionary
    lazada_soup_json = json.loads(response_text)

    for i in range(len(lazada_soup_json["mods"]["listItems"])):
        
        # Append product titles into list (Lazada)
        lazada_product_title_list.append(lazada_soup_json["mods"]["listItems"][i]['name'])
        
        # Append product price into list (Lazada)
        # Min price
        min_price = lazada_soup_json["mods"]["listItems"][i]['priceShow']
        trim = re.compile(r'[^\d.,]+')
        min_result = trim.sub('', min_price)
        lazada_product_price_min_list.append(min_result)
        
        try:
            # Max price
            max_price = lazada_soup_json["mods"]["listItems"][i]['originalPriceShow']
            if max_price is not None: 
                max_result = trim.sub('', max_price)
                lazada_product_price_max_list.append(max_result)
        except:
            lazada_product_price_max_list.append(min_result)

        # Append product review rating into list (Lazada)
        rating = lazada_soup_json["mods"]["listItems"][i]["ratingScore"].strip()
        if rating is not None and rating != "":
            ratingScoreFloat = round(float(rating), 2)
            lazada_product_star_list.append(ratingScoreFloat)
        else:
            lazada_product_star_list.append(0.00)
        
        # Append product review count into list (Lazada)
        # review = sum of both rating & review
        review = lazada_soup_json["mods"]["listItems"][i]["review"].strip()
        if review is None or review == "":
            lazada_product_rating_list.append(0)
        else: 
            lazada_product_rating_list.append(review)
        
        try:
            # Append product sold into list (Lazada)
            sold = lazada_soup_json["mods"]["listItems"][i]['itemSoldCntShow'].strip()
            if sold is not None: 
                trim_sold = trim.sub('', sold)
                final_sold = trim_sold.replace(",", "")
                lazada_product_sold_list.append(final_sold)
        except:
            lazada_product_sold_list.append(0)
        
        # Append product itemid into list (Lazada)
        lazada_product_itemid_list.append(lazada_soup_json["mods"]["listItems"][i]['itemId'])
        
        # Append product shopid into list (Lazada)
        lazada_product_shopid_list.append(lazada_soup_json["mods"]["listItems"][i]['sellerId'])
        
        # Append product itemurl into list (Lazada)
        itemurl = lazada_soup_json["mods"]["listItems"][i]['itemUrl']
        new_itemurl = itemurl.replace("//", "")
        lazada_product_itemurl_list.append(new_itemurl)

    lazada_data = {'product name':lazada_product_title_list,
        'minimum price':lazada_product_price_min_list,
        'maximum price':lazada_product_price_max_list,
        'average star rating':lazada_product_star_list,
        'total ratings':lazada_product_rating_list,
        'product sold': lazada_product_sold_list,
        'item id': lazada_product_itemid_list,
        'shop id': lazada_product_shopid_list,
        'product link url': lazada_product_itemurl_list}
    lazada_df_product = pd.DataFrame.from_dict(lazada_data)
    lazada_df_product.drop_duplicates(subset="product name", keep='first', inplace=True)

    return lazada_df_product

def scrape_shopee_info(searched_item: str) -> pd.DataFrame:
    """
        Scrape product information in first page of Shopee
        
        Parameters:
        searched_item: Item to request for
        
        Returns:
        A dataframe of search results for related products
    """
    # Create lists to store different product information
    shopee_product_title_list = list()
    shopee_product_price_min_list = list()
    shopee_product_price_max_list = list()
    shopee_product_star_list = list()
    shopee_product_rating_list = list()
    shopee_product_sold_list = list()
    shopee_product_itemid_list = list()
    shopee_product_shopid_list = list()
    shopee_product_itemurl_list = list()

    # Manual method of obtaining Request URL: F12 -> Network -> Fetch/XHR
    # Automate manual method by replacing keyword with searched_item
    shopee_url = 'https://shopee.sg/api/v4/search/search_items?by=relevancy&keyword=strawberry%20jam&limit=60&newest=0&order=desc&page_type=search&scenario=PAGE_GLOBAL_SEARCH&version=2'
    if ' ' in searched_item:
        shopee_searched_item = searched_item.replace(" ", "%20")
        new_shopee_url = shopee_url.replace('strawberry%20jam', shopee_searched_item)
    else:
        new_shopee_url = shopee_url.replace('strawberry%20jam', searched_item) 

    header = {
        "authority": "shopee.sg",
        "method": "GET",
        "path": "/api/v4/search/search_items?by=relevancy&keyword=chocolate%20chip%20cookies&limit=60&newest=0&order=desc&page_type=search&scenario=PAGE_GLOBAL_SEARCH&version=2&view_session_id=799fe51b-57b2-4b1f-b7e7-b51f569ae0d6",
        "scheme": "https",
        "accept": "application/json",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "af-ac-enc-dat": "AAcyLjUuMC0yAAABhR4oxVUAAAp6AhAAAAAAAAAAAuvlR3weVVU60ykHUkkzSmQs+0sol/82EyfDx/bVRcPaaRvYm7X6bHGNrNc63Dqg0j9+a4iWe63JyK9jolZNTpmwcUaeYJ097im09y3B7MjV04LMmpf/NhMnw8f21UXD2mkb2Js4fgyhyOmWOwejDo7KtGV7Oxy9zhjTB12xkp9B7tYU9MhC0RyJFeBNN5XYeE/3VhmiewNqu2Q/mfeJPvMM5zk870cdKImbM70WUPhYKLilVlJvNXiAl+E8XLaerB06qVaJ6i3phqNghZnCMVDn/mbSMXqGrJa39MtaNTrS3H1lOVRiZeaLJCVJU0SfG5dRLyVYTBnMjKJ10QOQ+FKU2Vymat4csEyeSv2AM9OK+6Gd123u5VelxX6+MyU5DquhpsgXX6erlx5qvwo+HrOJwVs56sRX8QFIk2CxAvw4jbU8pIABuTevMYrln0rStfUFCHXvS7lxehe+8KZzI1v+rhXJ8Kl+SyIUlhooEuCW2CzkkLTLlYtv/g73uV42QvDNWT7Pbo4u2oO1sW5K3SsIy+ebXLtQGxdA/+WLtMGmzqeqFdEcknYcoSs06Hl67Aq/+1bb16kx4gCDoX3UUg5taUDhkJTHnPLqBhzOmkYiCD/kP0gD9O1aYGgjmjuYbekv+r94rAESowEpdQeBx4X26VjY8p2ZqWP+6u+h8ZznD7de0nUNuR738DD1OUj403H5lwI=",
        "content-type": "application/json",
        "cookie": "REC_T_ID=f3f9baeb-4a49-11ed-8f9d-2cea7fad675a; SPC_T_ID=FTPJm+q2kyWgJPgVQPstR0JuQHm+kUZeBZXt4bhbmp0Qb9pXuiPj+Wf+nxrFbDoU9v2PcR807x9kpaM1Vsb2qnyuieswglzvn5gSXAm1XusOKUi5L6sUeB3ZSFT4VwoMV8iB80oGQ0lr0NlvA7UtCg==; SPC_T_IV=TTFaQVlCTU5XMGoxdEsxcQ==; SPC_F=G0ms3XhwCFdUW6pOFWl28my7C5cH7Xxh; _gcl_au=1.1.737366154.1665591702; _fbp=fb.1.1665591701676.1666866410; _tt_enable_cookie=1; _ttp=86b8c46a-feb6-4571-ae99-481e1dd2ee1c; SPC_IA=-1; SPC_EC=-; SPC_T_ID='FTPJm+q2kyWgJPgVQPstR0JuQHm+kUZeBZXt4bhbmp0Qb9pXuiPj+Wf+nxrFbDoU9v2PcR807x9kpaM1Vsb2qnyuieswglzvn5gSXAm1XusOKUi5L6sUeB3ZSFT4VwoMV8iB80oGQ0lr0NlvA7UtCg=='; SPC_U=-; SPC_T_IV='TTFaQVlCTU5XMGoxdEsxcQ=='; SPC_R_T_ID=FTPJm+q2kyWgJPgVQPstR0JuQHm+kUZeBZXt4bhbmp0Qb9pXuiPj+Wf+nxrFbDoU9v2PcR807x9kpaM1Vsb2qnyuieswglzvn5gSXAm1XusOKUi5L6sUeB3ZSFT4VwoMV8iB80oGQ0lr0NlvA7UtCg==; SPC_R_T_IV=TTFaQVlCTU5XMGoxdEsxcQ==; cto_bundle=ucRttF82NnB5TWVyNG5UTUVCMyUyRnFvUE5RQkcxaDlqcmlzUlRIUW13JTJCWUxWNjAlMkIxS2UlMkZkaEglMkJ4T3BMSVk0RVpaRXc1WTVlcmZWMDVIQTJXJTJGQ3IycDJlSWF4YlUwdVhJNWtQMjlMM04wVmJjcEtyRWhMY1FUc1Vuc1Z3JTJCR3pJMkhjbVhKRUNGSTc0dDFSSjV5JTJGYlFzcVJDSWZBJTNEJTNE; SPC_SI=ssWNYwAAAABvcVBTNHdtNY8eOwMAAAAAeFhXTFY4ZVM=; _gid=GA1.2.368919447.1670954951; __LOCALE__null=SG; csrftoken=chTMgA70WVGtIprdE4Rsauuh6n4GLOIo; AMP_TOKEN=%24NOT_FOUND; _QPWSDCXHZQA=a2ce6d4b-7435-4715-93b6-fe133da95aa8; shopee_webUnique_ccd=LGVTrssQp0nbELX6dtaHPw%3D%3D%7C4F1pro3Lk6E47ceS2407l0AvqoK2q8ZGaZRfVbolg2vuymSDkkv4KrOAI5DGR1Sl56VrGbAGpDQnbbKRhasPqrVWtrYRSdGRJSM%3D%7Cs7A%2FyNtuk11oVVVL%7C06%7C3; ds=f724f103f5973105f1f5aedc0dd374b2; _ga=GA1.1.1695583839.1665591703; _dc_gtm_UA-61921742-7=1; _ga_4572B3WZ33=GS1.1.1671250163.29.1.1671250201.22.0.0",
    }

    # Make requests to website
    response = requests.get(new_shopee_url, headers=header)
    response_text = response.text

    # Convert json to Python dictionary
    shopee_soup_json = json.loads(response_text)

    for i in range(len(shopee_soup_json["items"])):
        
        # Append product titles into list (Shopee)
        product_title = shopee_soup_json["items"][i]['item_basic']['name']
        if ' ' in product_title:
            producttitle = product_title.replace(" ", "-")
        else:
            producttitle = product_title
        shopee_product_title_list.append(shopee_soup_json["items"][i]['item_basic']['name'])
        
        # Append product price into list (Shopee)
        if shopee_soup_json["items"][i]['item_basic']['price_min'] == shopee_soup_json["items"][i]['item_basic']['price_max']:
            shopee_product_price_min_list.append(shopee_soup_json["items"][i]['item_basic']['price_max']/100000)
            shopee_product_price_max_list.append(shopee_soup_json["items"][i]['item_basic']['price_max']/100000)
        else:        
            shopee_product_price_min_list.append(shopee_soup_json["items"][i]['item_basic']['price_min']/100000)
            shopee_product_price_max_list.append(shopee_soup_json["items"][i]['item_basic']['price_max']/100000)
        
        # Append product review rating into list (Shopee)
        shopee_product_star_list.append(round(shopee_soup_json["items"][i]["item_basic"]["item_rating"]["rating_star"],2))
            
        # Append product review count into list (Shopee)
        # ratingSum = must have both rating + review
        ratingArr = shopee_soup_json["items"][i]["item_basic"]["item_rating"]["rating_count"]
        ratingSum = ratingArr[0]
        shopee_product_rating_list.append(ratingSum)
        
        # Append product itemid into list (Shopee)
        itemid = shopee_soup_json["items"][i]['item_basic']['itemid']
        shopee_product_itemid_list.append(shopee_soup_json["items"][i]['item_basic']['itemid'])
        
        # Append product shopid into list (Shopee)
        shopid = shopee_soup_json["items"][i]['item_basic']['shopid']
        shopee_product_shopid_list.append(shopee_soup_json["items"][i]['item_basic']['shopid'])
        
        # Append product sold into list (Shopee)
        sold_cnt = shopee_soup_json["items"][i]['item_basic']['historical_sold']
        shopee_product_sold_list.append(sold_cnt)
        
        # Append product itemurl into list (Shopee)
        shopee_itemurl_template = "https://shopee.sg/{producttitle}-i.{shopid}.{itemid}?sp_atk=402d9080-de98-452d-bf57-aac87d2fb5e1&xptdk=402d9080-de98-452d-bf57-aac87d2fb5e1"
        shopee_itemurl = shopee_itemurl_template.format(producttitle=producttitle, itemid=itemid, shopid=shopid)
        shopee_product_itemurl_list.append(shopee_itemurl)

    shopee_data = {'product name':shopee_product_title_list,
        'minimum price':shopee_product_price_min_list,
        'maximum price':shopee_product_price_max_list,
        'average star rating':shopee_product_star_list,
        'total ratings': shopee_product_rating_list,
        'product sold': shopee_product_sold_list,
        'item id': shopee_product_itemid_list,
        'shop id': shopee_product_shopid_list,
        'product link url': shopee_product_itemurl_list}
    shopee_df_product = pd.DataFrame.from_dict(shopee_data)
    shopee_df_product.drop_duplicates(subset="product name", keep='first', inplace=True)

    return shopee_df_product

def web_scrape_product_info(searched_item: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrapes product information from both Lazada and Shopee

    Args:
        searched_item (str): Item to scrape for

    Returns:
        Tuple[pd.Dataframe, pd.DataFrame]: Returns a tuple dataframe for lazada and shopee. Output: (Lazada, Shopee)
    """
    lazada_df_product = scrape_lazada_info (searched_item)
    shopee_df_product = scrape_shopee_info (searched_item)
    return lazada_df_product, shopee_df_product

def compare_similarity(df: pd.DataFrame, searched_item_str: str) -> pd.DataFrame:
    """
    Compare similarity between searched item and returned product listings

    Args:
        df (pd.DataFrame): Dataframe to perform similary rating on
        searched_item_str (str): String to perform similarity rating on

    Returns:
        pd.DataFrame: Returns the dataframe with a new column ['similarity_ratio']. Indicates % of similarity
    """
    title_ratio_list = []
    column_index = 0
    df_length = len(df)

    for row_index in range (df_length):
        platform_title_str = df.iloc[row_index, column_index]
        platform_title_str = platform_title_str.lower() 
        ratio = fuzz.partial_ratio(searched_item_str, platform_title_str)
        title_ratio_list.append(ratio)
        
    df['similarity_ratio'] = title_ratio_list    
    return df

def filter_search(df: pd.DataFrame, searched_item: str) -> pd.DataFrame:
    """
    Filter returned search results from a Platform

    Args:
        df (pd.DataFrame): Dataframe to filter search resutls
        searched_item (str): Input used to filter search results

    Returns:
        pd.DataFrame: Dataframe that has been filtered to select only that passes the similarity checks
    """
    # Convert searched product string to lowercase
    searched_item_str = searched_item.lower()
        
    # Apply compare_simlarity function to existing Lazada dataframe
    filtered_product_df = compare_similarity(df, searched_item_str)

    # Dataframe after filtering
    first_filtered_product_df = filtered_product_df[filtered_product_df['similarity_ratio'] >= 80]

    # Drop product listings with 0 reviews
    first_filtered_product_df.drop(first_filtered_product_df[(first_filtered_product_df['total ratings'] == 0)].index, inplace=True)

    return first_filtered_product_df

def get_review_lazada(item_id: str) -> pd.DataFrame:
    """
    Scrape customer reviews of a single product listing (Lazada)

    Args:
        item_id (str): Item ID of the product

    Returns:
        pd.DataFrame: A dataframe of the reviews found on that product page
    """
    # Manual method of obtaining Request URL: F12 -> Network -> Fetch/XHR
    # Automate manual method by replacing keyword with searched_item
    lazada_product_link_url2 = 'https://my.lazada.sg/pdp/review/getReviewList?itemId={item_id}&pageSize=50&filter=0&sort=0&pageNo=1'
    
    # Create empty list to hold lists of all individual product listing reviews 
    df_review2 = []
    
    header = {
        "authority": "member.lazada.sg",
        "method": "GET",
        "path": "/user/api/getUser",
        "scheme": "https",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9,ms;q=0.8",
        "cookie": "client_type=desktop; lzd_cid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; t_uid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; lzd_sid=115065e370eb173c150b89b82f3657c2; anon_uid=e0abc68e7864a7414fce4c16e60580b4; _tbtoken=eee3b383373e0; t_fv=1668525165789; _gcl_au=1.1.1770287619.1668525166; hng=SG|en-SG|SGD|702; hng.sig=ryBKXOqZIsp9xOQ3YsZRgD7f-p0UaGB2pZ4BbZM8uEc; xlly_s=1; _m_h5_tk=7b313bd7db4ef975bc78b67fc7d4643e_1672938834121; _m_h5_tk_enc=a41920d8e3af937201c0f0245c75f9eb; t_sid=0SuQOSPKsOv0mJcSGimyZBJO0cvmrGzm; utm_channel=NA; x5sec=7b22617365727665722d6c617a6164613b32223a223836376635353730373131383033663036333263643633666662633339313131434d373932353047454d66452f5a693367706a454444434b6d5a6e332b2f2f2f2f2f384251414d3d227d; isg=BHZ2n7xICXfCuP1WvJP7BSCIx6x4l7rRJmHRO-BcINn0Ixe9SyWu4BuaO_dPi7Lp; l=fBOgmYC7T1fHvQtZBO5Courza779EQOb8sPzaNbMiIEGa6NG9FZdHNCFz5UeWdtj_T55-e-zEHwxcdeMWmaU-jkDBeYQ-JDocaJw-ewZ_o7d.; tfstk=cuNdBr1l-GjnM1So_XBMaYJ70rQGaUj-feikwSOwxPjMaGdkasbuiS3Xj9gS0DQO.",
        "dnt": "1",
        "sec-ch-ua": '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    }
    
    try : 
        lazada_comment_url2 = lazada_product_link_url2.format(item_id=item_id) 
        lazada_json_response2 = requests.get(lazada_comment_url2, headers=header).text
        indi_product2 = json.loads(lazada_json_response2)
        
    except:
        print(f"Error on Product {item_id}, request resopnse: {lazada_json_response2}")
        
    # Check if product has reviews or not, if yes
    if indi_product2["model"]["items"] is not None:
       
        for i in range(len(indi_product2["model"]["items"])):
            # Check if there is comment in review
            # Append empty list into df_review if there is no comment
            if indi_product2["model"]["items"][i]['reviewContent'] is None:
                proper_review2 = []
                df_review2.append(proper_review2)
            
            # If there is comment in review
            # Append comments into individual proper_review -> df_review
            else:
                proper_review2 = []
                proper_review2.append(indi_product2["model"]["items"][i]['buyerName'])
                proper_review2.append(indi_product2["model"]["items"][i]['rating'])
                proper_review2.append(indi_product2["model"]["items"][i]['reviewContent'])
                df_review2.append(proper_review2)
                
    else:
        # Append empty list into df_review if there is no review
        proper_review2 = []
        df_review2.append(proper_review2)

    return df_review2

def get_review_shopee(shop_id: str, item_id: str) -> pd.DataFrame:
    """
    Scrape customer reviews of a single product listing (Shopee)

    Args:
        shop_id (str): Shop Id of product in Shopee
        item_id (str): Item Id of product to scrape

    Returns:
        pd.DateFrame: A dataframe of the reviews found on that product page
    """
    # Manual method of obtaining Request URL: F12 -> Network -> Fetch/XHR
    # Automate manual method by replacing keyword with searched_item
    shopee_product_link_url = 'https://shopee.sg/api/v2/item/get_ratings?filter=1&flag=1&itemid={item_id}&limit=50&offset=0&shopid={shop_id}&type=0'

    # Create empty list to hold lists of all individual product listing reviews 
    df_review = []
    
    header = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "cookie": "client_type=desktop; lzd_cid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; t_uid=faa4a6ce-e35f-4e71-af42-6139a2153ebc; lzd_sid=115065e370eb173c150b89b82f3657c2; anon_uid=e0abc68e7864a7414fce4c16e60580b4; _tbtoken=eee3b383373e0; t_fv=1668525165789; _gcl_au=1.1.1770287619.1668525166; hng=SG|en-SG|SGD|702; hng.sig=ryBKXOqZIsp9xOQ3YsZRgD7f-p0UaGB2pZ4BbZM8uEc; xlly_s=1; t_sid=l8sXsB19lXMUd4dLs18bmLiSB3YHtmDx; utm_channel=NA; _m_h5_tk=191dd27999507773d60d45d499d37214_1672696167776; _m_h5_tk_enc=3b6fc543579b2798223b8d62b1a0a25f; tfstk=cgt5Bvcel7V5ZSn3q_MqU_R-YrjhZq21d41kFbHHcPFsV1v5iGrN1E8H-WIFpt1..; l=eBOgmYC7T1fHvFsMKOfZhurza77OSIRAWuPzaNbMiOCP_HfH568AW67cGR8MC3GNh6DJR3ykA-nXBeYBqQAonxvOTPQwTBkmn; isg=BFpa8tnS7eayqGGKOB9fEexcqwB8i95lerVtB2TTBu241_oRTBsudSAhp6vLSlb9",
        "referer": "https://www.lazada.sg/",
        "authority": "member.lazada.sg",
        "sec-ch-ua": '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"'
    }
    
    # Make request to website
    shopee_comment_url = shopee_product_link_url.format(shop_id=shop_id, item_id=item_id)  
    shopee_json_response = requests.get(shopee_comment_url, headers=header).text
    indi_product = json.loads(shopee_json_response)
    
    if indi_product["data"]["ratings"] is not None:
        # Append comments into individual proper_review -> df_review
        for i in range(len(indi_product["data"]["ratings"])):
            proper_review = []
            proper_review.append(indi_product["data"]["ratings"][i]['author_username'])
            proper_review.append(indi_product["data"]["ratings"][i]['rating_star'])
            proper_review.append(indi_product["data"]["ratings"][i]['comment'])
            df_review.append(proper_review)
    else:
        # Append empty list into df_review if there is no comment
        proper_review = []
        df_review.append(proper_review)
        
    return df_review

def decontracted(review_text: str) -> str:
    """
    Function to remove contractions

    Args:
        review_text (str): Review text to decontract

    Returns:
        str: Decontracted string of input
    """
    # specific
    review_text = re.sub(r"won\'t", "will not", str(review_text))
    review_text = re.sub(r"can\'t", "can not", str(review_text))

    # general
    review_text = re.sub(r"n\'t", " not", str(review_text))
    review_text = re.sub(r"\'re", " are", str(review_text))
    review_text = re.sub(r"\'s", " is", str(review_text))
    review_text = re.sub(r"\'d", " would", str(review_text))
    review_text = re.sub(r"\'ll", " will", str(review_text))
    review_text = re.sub(r"\'t", " not", str(review_text))
    review_text = re.sub(r"\'ve", " have", str(review_text))
    review_text = re.sub(r"\'m", " am", str(review_text))

    return review_text

def detokenize (to_detokenize: str) -> List[str]:
    """
    Function to detokenize lemmatized tokens

    Args:
        to_detokenize (str): String to detokenize

    Returns:
        List[str]: list of detokenized tokens
    """
    detokenized_list = " "
    for tokens in to_detokenize:
        detokenized_list = " ".join(str(tokens) for tokens in to_detokenize)
    return detokenized_list

def review_preprocessing(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess scraped reviews

    Args:
        review_df (pd.DataFrame): Dataframe of Reviews to preprocess

    Returns:
        pd.DataFrame: Preprocessed Dataframe of Reviews
    """
    
    stop = text.ENGLISH_STOP_WORDS
    ENGLISH_STOP_WORDS = set(stopwords.words('english')).union(set(stop))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Remove space & convert to lowercase
    review_df['cleaned review'] = review_df['review'].str.strip().str.lower()

    # Remove contractions
    review_df['cleaned review'] = review_df['cleaned review'].apply(decontracted)
    
    # Remove punctuation
    review_df['cleaned review'] = review_df['cleaned review'].str.replace('[^\w\s]',' ', regex=True)
    
    # Remove numbers
    review_df['cleaned review'] = review_df['cleaned review'].str.replace('\d+', '', regex=True)
    
    # Remove stopwords
    review_df['cleaned review'] = review_df['cleaned review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOP_WORDS)]))
    
    # Tokenize reviews
    review_df['cleaned review'] = review_df.apply(lambda row: nltk.word_tokenize(row['cleaned review']), axis=1)
    
    # Perform lemmatization
    review_df['cleaned review'] = review_df['cleaned review'].apply(lambda lst:[lemmatizer.lemmatize(word) for word in lst])
    
    # Create new column for detokenized reviews
    review_df['cleaned review'] = review_df['cleaned review'].apply(detokenize)

    return review_df

def review_prediction(product_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Function to predict sentiments

    Args:
        product_reviews (pd.DataFrame): Dataframe of Reviews to predict on

    Returns:
        pd.DataFrame: Dataframe with a new column ['sentiment] that shows the prediction output for the review
    """
    
    # Load vectorizer & model
    loaded_vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))
    loaded_model = pickle.load(open("NBmodel.pickle", "rb"))
    
    tbp_review = product_reviews['cleaned review'].to_numpy()
    vectorized = loaded_vectorizer.transform(tbp_review)
    predicted = loaded_model.predict(vectorized)
    product_reviews['sentiment'] = predicted
        
    return product_reviews

def prep_scraped_reviews(total_indi_products: List[Tuple[str, str, str]]) -> List[pd.DataFrame]:
    """
    Function to append cleaned & predicted reviews into dataframe

    Args:
        total_indi_products (List[Tuple[str, str, str]]): List of Reviews in the format of ['user', 'rating', 'review']

    Returns:
        List[pd.DataFrame]: Returns a list of dataframes. Each dataframe is a product and includes their reviews inside it
    """

    df_list = []
    
    for product in total_indi_products: 
        # remove empty list in total_indi_products
        filtered_product = [review for review in product if review]

        if len(filtered_product) != 0:
            review_df = pd.DataFrame(filtered_product[0:], columns = ['user', 'rating', 'review'])

            # call review_preprocessing function: preprocess review + add new column "cleaned review"
            product_review = review_preprocessing(review_df)

            # call review_prediction function: run model on "cleaned review" + add new column "sentiment"
            predicted_review = review_prediction(product_review)
            
            # append predicted df to list
            df_list.append(predicted_review)
        else:
            review_df = pd.DataFrame({'user': ["na"], 'rating': [0], 'review': ["na"], 'cleaned review': ["na"], 'sentiment': ["invalid"]})
            df_list.append(review_df)

    return df_list

def compute_sentiment_stats (list_dataframes_reviews: List[pd.DataFrame], platform_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute sentiment stats

    Args:
        list_dataframes_reviews (List[pd.DataFrame]): A list of dataframes. Dataframes includes reviews on each product
        platform_df (pd.DataFrame): Platform Dataframe to attach statistics of reviews to

    Returns:
        pd.DataFrame: platform_df with new columns ['positive reviews', 'negative reviews', 'invalid reviews']
    """

    pos_arr = []
    neg_arr = []
    inv_arr = []

    for product in list_dataframes_reviews:
        # calculate number of positive and negative reviews
        total_positive = sum(product['sentiment'] == 'positive')
        total_negative = sum(product['sentiment'] == 'negative')
        total_invalid = sum(product['sentiment'] == 'invalid')

        pos_arr.append(total_positive)
        neg_arr.append(total_negative)
        inv_arr.append(total_invalid)

    platform_df['positive reviews'] = pos_arr
    platform_df['negative reviews'] = neg_arr
    platform_df['invalid reviews'] = inv_arr

    return platform_df

def summarised_prediction(total_indi_products: List[Tuple[str, str, str]], platform_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Function to summarise prediction results

    Args:
        total_indi_products (List[Tuple[str, str, str]]): List of Reviews in the format of ['user', 'rating', 'review']
        platform_df (pd.DataFrame): Dataframe of all available products in a platform

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
        (final_platform_output, final_platform_df)\n
        final_platform_output: a list of dataframes. Each dataframe is a product and includes their reviews inside it\n
        final_platform_df: platform_df with new columns ['positive reviews', 'negative reviews', 'invalid reviews']

    """
    final_platform_output = prep_scraped_reviews(total_indi_products)
    final_platform_df = compute_sentiment_stats(final_platform_output, platform_df)

    return final_platform_output, final_platform_df

def merge_dataframes(final_lazada_df : pd.DataFrame, final_shopee_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create final dataframes

    Args:
        final_lazada_df (pd.DataFrame): final dataframe for lazada
        final_shopee_df (pd.DataFrame): final dataframe for shopee

    Returns:
        pd.DataFrame: Merged dataframe of lazada and shopee with new column ['platform'] that dictates which platform product belongs to
    """
    # Set new platform columns 
    # For Lazada
    final_lazada_df.insert(0, 'platform', 'Lazada')

    # For Shopee
    final_shopee_df.insert(0, 'platform', 'Shopee')

    # Rename columns (Lazada)
    final_lazada_df.rename(columns = {'similarity_ratio':'relevance score',
                                    'scraped_review':'scraped review'}, inplace = True)

    # Rename columns (Shopee)
    final_shopee_df.rename(columns = {'similarity_ratio':'relevance score',
                                    'scraped_review':'scraped review'}, inplace = True)

    # Merge dataframes together
    dfs = [final_lazada_df, final_shopee_df]
    final_df = pd.concat(dfs)

    return final_df

def review_long_df(final_platform_output: List[pd.DataFrame], final_platform_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create long dataframes

    Args:
        final_platform_output (List[pd.DataFrame]): A list of dataframes. Dataframes includes reviews on each product
        final_platform_df (pd.DataFrame): A platform_df with columns ['positive reviews', 'negative reviews', 'invalid reviews']

    Returns:
        pd.DataFrame: dataframe in long format
    """

    
    platform_long_product = pd.DataFrame(columns=["product", "positive reviews", "negative reviews", "rating", "review", "sentiment"])
    index = 0
    for product in final_platform_output:
        product_title = final_platform_df['product name'].iloc[index]
        product_title_ar = [product_title] * len(product)
        
        positive_count = final_platform_df['positive reviews'].iloc[index]
        positive_count_ar = [positive_count] * len(product)
        
        negative_count = final_platform_df['negative reviews'].iloc[index]
        negative_count_ar = [negative_count] * len(product)
        
        product_df = final_platform_output[index].drop(['user', 'cleaned review'], axis=1)
        product_df['product'] = product_title_ar
        product_df['positive reviews'] = positive_count_ar
        product_df['negative reviews'] = negative_count_ar
        platform_long_product = pd.concat([platform_long_product, product_df])
        index += 1

    # Remove rows with invalid sentiments
    platform_long_product = platform_long_product[(platform_long_product.sentiment != "invalid")]   

    return platform_long_product

def remove_invalid_reviews(final_platform_df : pd.DataFrame) -> pd.DataFrame:
    """
    Removes products with zero reviews therefore having invalid sentiments

    Args:
        final_platform_df (pd.DataFrame): Platform dataframe for each platform

    Returns:
        pd.DataFrame: Cleansed Dataframe of review
    """
    # Removing Invalid Reviews
    final_platform_df.drop(final_platform_df[(final_platform_df['invalid reviews'] != 0)].index, inplace=True)
    final_platform_df = final_platform_df.drop(['invalid reviews'], axis=1)
    return final_platform_df

def get_wordcloud_review(platform_review_df : pd.DataFrame) -> pd.DataFrame:
    """
    Function to get individual reviews for word clouds

    Args:
        platform_review_df (pd.DataFrame): Dataframe of platforms reviews on a certain type of product

    Returns:
        pd.DataFrame: Dataframe of wordcloud words
    """
    stop = text.ENGLISH_STOP_WORDS
    ENGLISH_STOP_WORDS = set(stopwords.words('english')).union(set(stop))

    # Remove space & convert to lowercase
    platform_review_df['wordcloud review'] = platform_review_df['review'].str.strip().str.lower()

    # Remove contractions`
    platform_review_df['wordcloud review'] = platform_review_df['wordcloud review'].apply(decontracted)

    # Remove punctuation
    platform_review_df['wordcloud review'] = platform_review_df['wordcloud review'].str.replace('[^\w\s]',' ', regex=True)

    # Remove numbers
    platform_review_df['wordcloud review'] = platform_review_df['wordcloud review'].str.replace('\d+', '', regex=True)

    # Remove stopwords
    platform_review_df['wordcloud review'] = platform_review_df['wordcloud review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOP_WORDS)]))
    
    # Tokenize reviews
    platform_review_df['wordcloud review'] = platform_review_df['wordcloud review'].apply(lambda row: nltk.word_tokenize(row))

    # Flatten review list
    platform_review_list = platform_review_df["wordcloud review"].values.tolist()
    flat_review_list = []
    for review in platform_review_list:
        for token in review:
            flat_review_list.append(token)

    # Create a new dataframe
    wordcloud_review_df = pd.DataFrame(columns=["review"])
    wordcloud_review_df.at[0, 'review'] = flat_review_list

    # Convert from list to string
    flat_review_string = ''
    for token in flat_review_list:
        flat_review_string += " " + token

    # Update column value
    wordcloud_review_df.at[0, 'review'] = flat_review_string
    
    return wordcloud_review_df

def wordcloud_review_df(platform_long_product : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to create dataframes for review word clouds

    Args:
        platform_long_product (pd.DataFrame): Dataframe in long format to grab wordcloud words from

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (platform_pos_review, platform_neg_review)\n
        platform_pos_review: Positive words in platform
        platform_neg_review: Negative words in platform
    """

    # Get positive Lazada reviews
    platform_get_pos_review = platform_long_product['sentiment'].values == "positive"
    platform_get_neg_review = platform_long_product['sentiment'].values == "negative"

    # Select rows of dataframes of different sentiments
    platform_pos_review_df = platform_long_product[platform_get_pos_review]
    platform_neg_review_df  = platform_long_product[platform_get_neg_review]

    platform_pos_review = get_wordcloud_review(platform_pos_review_df)
    platform_neg_review = get_wordcloud_review(platform_neg_review_df)

    return platform_pos_review, platform_neg_review