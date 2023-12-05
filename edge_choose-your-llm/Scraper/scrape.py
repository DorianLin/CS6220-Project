import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import traceback

def get_json_format_data():
    url = 'https://huggingfaceh4-open-llm-leaderboard.hf.space/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser') # response.content is in raw bytes
    script_elements = soup.find_all('script')
    json_format_data = json.loads(str(script_elements[1])[31:-10])
    return json_format_data


def get_datas(data):
    data = data['components'] # list
    result_list = []
    headers_len = 0
    component_index = 0 # 0-indexed list index
    for i in range(len(data)):
        if not data[i]['type'] == 'dataframe':
            continue
        headers = data[i]['props']['value']['headers'] # list of strings
        if len(headers) > headers_len:
            headers_len = len(headers)
            component_index = i
    # print(component_index, headers_len)

    headers = data[component_index]['props']['value']['headers']
    data = data[component_index]['props']['value']['data']
    assert len(headers) == len(data[0])
    for i in range(len(data)):
        results = data[i]
        results_json = {}
        for idx, header in enumerate(headers):
            if header == 'Model':
                html_content = results[idx]
                soup = BeautifulSoup(html_content, 'html.parser')
                links = soup.find_all('a')
                if len(links) >= 2:
                    model_repo = links[0].get('href')
                    model_experiment_details = links[1].get('href')
                else:
                    model_repo = ''
                    model_experiment_details = ''
                results_json['Model_repo'] = model_repo
                results_json['Model_experiment_details'] = model_experiment_details
            else:
                results_json[header] = results[idx]
        result_list.append(results_json)
    
    return result_list

def get_datas_legacy(data):
    # print(data)
    for component_index in range(10, 50, 1): # component_index sometimes changes when they update the space, we can use this "for" loop to avoid changing component index manually
        try:
            result_list = []
            i = 0
            while True:
                try:
                    results = data['components'][component_index]['props']['value']['data'][i]
                    # print(results)
                    # Parse the HTML content to extract links
                    html_content = results[1]
                    soup = BeautifulSoup(html_content, 'html.parser')
                    links = soup.find_all('a')
                    if len(links) >= 2:
                        model_repo = links[0].get('href')
                        model_experiment_details = links[1].get('href')
                    else:
                        model_repo = ''
                        model_experiment_details = ''

                    try:
                        results_json = {
                            "Type": results[10],
                            "Model_repo": model_repo,
                            "Model_experiment_details": model_experiment_details,
                            "Average": results[2],
                            "ARC": results[3],
                            "HellaSwag": results[4],
                            "MMLU": results[5],
                            "TruthfulQA": results[6],
                            "Winogrande": results[7],
                            "GSM8K": results[8],
                            "DROP": results[9],
                            "Architecture": results[11],
                            "Precision": results[12],
                            "Hub_License": results[13],
                            "#Params (B)": results[14],
                            "Hub_like": results[15],
                            "Available_on_the_hub": results[16],
                            "Model_sha": results[17],
                            "Model_name_for_query": results[18]
                        }
                    except IndexError: # wrong component_index, break current while loop
                        # print("1")
                        break
                    result_list.append(results_json)
                    i += 1
                except IndexError: # No rows to extract so return the list (We know it is the right component index because we didn't break out of loop on the other exception.)
                    # print(component_index, "2")
                    return result_list
        except (KeyError, TypeError) as e: # wrong component_index, proceed to next one
            # print("3")
            continue

    return result_list

def scrape():
    parser = argparse.ArgumentParser(description="Scrape and export data from the Hugging Face leaderboard")
    parser.add_argument("-csv", action="store_true", help="Export data to CSV")
    parser.add_argument("-html", action="store_true", help="Export data to HTML")
    parser.add_argument("-json", action="store_true", help="Export data to JSON")
    
    args = parser.parse_args()

    data = get_json_format_data()
    # Save the dictionary to a text file
    # with open('leaderboard_json.txt', 'w') as file:
    #     json.dump(data, file)

    # with open('components.txt', 'w') as file:
    #     for item in data['components']:
    #         file.write("%s\n\n\n\n\n\n" % item)

    # df_json = json.dumps(data['components'][37])
    # with open('leaderboard_df.txt', 'w') as file:
    #     json.dump(df_json, file)

    # print(len(data['components']))
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)

    if not args.csv and not args.html and not args.json:
        args.csv = True  # If no arguments are provided, default to CSV export

    if args.csv:
        df.to_csv("open-llm-leaderboard.csv", index=False)
        print("Leaderboard data exported to CSV")

    if args.html:
        df.to_html("open-llm-leaderboard.html", index=False)
        print("Leaderboard data exported to HTML")

    if args.json:
        df.to_json("open-llm-leaderboard.json", orient='records')
        print("Leaderboard data exported to JSON")

if __name__ == "__main__":
    scrape()