import os
from firecrawl import Firecrawl
from logs import get_logger
import pypdf
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

textSplitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
logging = get_logger(__name__)

crawl=Firecrawl(api_key="fc-2b841a657e054ce3a321c019e4692013")

currdir = os.path.dirname(os.path.abspath(__file__))
logging.info("Current Directory: %s", currdir)
dataDir = os.path.join(currdir, 'data')
logging.info("Data Directory: %s", dataDir)

if not os.path.exists(dataDir):
    os.makedirs(dataDir)
    logging.info("Created data directory at: %s", dataDir)

def getFileLists(filePath):
    pdfFiles = [f for f in os.listdir(filePath) if f.endswith('.pdf')]
    csvFiles = [f for f in os.listdir(filePath) if f.endswith('.csv')]
    txtFiles = [f for f in os.listdir(filePath) if f.endswith('.txt')]
    logging.info("Found %d PDF files, %d CSV files, %d TXT files", len(pdfFiles), len(csvFiles), len(txtFiles))
    return pdfFiles, csvFiles, txtFiles
    
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
        # print()
        content.append(response.json["theoryContent"])
    
def AllContent():
    # Load web content first
    linkContentLoader(links)
    
    # Get file lists
    pdfFiles, csvFiles, txtFiles = getFileLists(dataDir)
    
    allchunks = []
    allTextContent = []
    
    # Process PDF files
    if pdfFiles:
        for pdfFile in pdfFiles:
            try:
                reader = pypdf.PdfReader(os.path.join(dataDir, pdfFile))
                pdfText = ""
                for page in reader.pages:
                    pageText = page.extract_text()
                    if pageText.strip():
                        pdfText += pageText + "\n"
                if pdfText.strip():
                    allTextContent.append(pdfText)
                logging.info("Processed PDF: %s", pdfFile)
            except Exception as e:
                logging.error("Error processing PDF %s: %s", pdfFile, str(e))
    
    # Process CSV files
    if csvFiles:
        for csvFile in csvFiles:
            try:
                df = pd.read_csv(os.path.join(dataDir, csvFile))
                csvText = df.to_string(index=False)
                if csvText.strip():
                    allTextContent.append(csvText)
                logging.info("Processed CSV: %s", csvFile)
            except Exception as e:
                logging.error("Error processing CSV %s: %s", csvFile, str(e))
    
    # Process TXT files
    if txtFiles:
        for txtFile in txtFiles:
            try:
                with open(os.path.join(dataDir, txtFile), 'r', encoding='utf-8') as f:
                    txtContent = f.read()
                    if txtContent.strip():
                        allTextContent.append(txtContent)
                logging.info("Processed TXT: %s", txtFile)
            except Exception as e:
                logging.error("Error processing TXT %s: %s", txtFile, str(e))
    
    # Process web content
    if content:
        for webContent in content:
            # print(webContent)
            allTextContent.append(webContent)
        logging.info("Processed web content")
    
    # Create chunks from all text content
    if allTextContent:
        allchunks = textSplitter.create_documents(allTextContent)
        logging.info("Created %d chunks from all content", len(allchunks))
    else:
        logging.warning("No content found to process")
    
    logging.info("Loaded all files from data directory.")
    return allchunks

def main():
    allchunks = AllContent()
    logging.info("Created %d chunks from all files.", len(allchunks))
    return allchunks

if __name__ == "__main__":
    main()