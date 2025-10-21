from firecrawl import Firecrawl
crawl=Firecrawl(api_key="fc-2b841a657e054ce3a321c019e4692013")  
links=["https://www.geeksforgeeks.org/deep-learning/fine-tuning-using-lora-and-qlora/"]
content=[]
def linkContentLoader(links:list):
    for link in links:
        response = crawl.scrape(
            url=link,
            formats=[{
            "type": "json",
            "prompt": "Extract the all theory content from the article"
            }],
            only_main_content=False,
            timeout=120000
        )
        print(response.json["theoryContent"])
        content.append(response.json)
linkContentLoader(links)
allTextContent=[]
