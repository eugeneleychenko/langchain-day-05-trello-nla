from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from trello import TrelloClient
from bs4 import BeautifulSoup
# from langchain.document import Document

API_KEY = os.getenv("API_KEY")
TOKEN = os.getenv("TOKEN")
openai_api_key=os.getenv("openai_api_key")


def get_trello_boards(client):
    return client.list_boards()

def get_trello_cards(board):
    return board.list_cards()

def get_list_from_card(board, card):
    return board.get_list(card.list_id)

def get_members_from_card(api_key, token, card_id):
    return TrelloClient(api_key=api_key, token=token).get_card_members(card_id)

def main():
    api_key = API_KEY
    token = TOKEN

    client = TrelloClient(
        api_key=api_key,
        token=token
    )

    boards = get_trello_boards(client)
    board_names = [board.name for board in boards]
    ### hard coded board name
    # selected_board_name = input('Enter a board name: ')
    selected_board_name = "Smith Street Enterprises"

    for board in boards:
        if board.name == selected_board_name:
            selected_board = board
            break

    cards = get_trello_cards(selected_board)
    documents = []  # List to store Document objects
    for card in cards:
        text_content = ""
        if card.name:
            text_content = "Title: " + card.name + "\n"
        if card.description.strip():
            text_content += "Description: " + BeautifulSoup(card.description, "lxml").get_text() + "\n"
        list = get_list_from_card(selected_board, card)
        text_content += "List: " + list.name + "\n"
        members = get_members_from_card(api_key, token, card.id)
        text_content += "Members: " + ', '.join([member['fullName'] for member in members]) + "\n"
        for checklist in card.checklists:
            if checklist.items:
                items = [
                    f"{item['name']}:{item['state']}" for item in checklist.items
                ]
                text_content += "Checklist: " + checklist.name + "\n" + "\n".join(items) + "\n"
        comments = [
            BeautifulSoup(comment["data"]["text"], "lxml").get_text()
            for comment in card.comments
        ]
        text_content += "Comments: " + "\n".join(comments) + "\n"
        metadata = {
            "title": card.name,
            "id": card.id,
            "url": card.url,
        }
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)  # Add the Document object to the list

    return documents  # Return the list of Document objects

def write_output_to_file(output):
    # Open the file in write mode
    file = open("output.txt", "w")

    # Write each element of the list to the file
    for item in output:
        file.write(str(item) + "\n")

    # Close the file
    file.close()

def run_query():
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo")

    loader = TextLoader('output.txt')
    doc = loader.load()
    print(f"You have {len(doc)} document")
    print(f"You have {len(doc[0].page_content)} characters in that document")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    docs = text_splitter.split_documents(doc)

    # Get your embeddings engine ready
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
    docsearch = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    query = "Which projects have not yet been started?"
    return qa.run(query)

# Call the main function and store the result
documents = main()

# Now you can use the 'documents' variable in another section of your notebook
output = [(doc.page_content, doc.metadata) for doc in documents]
print(output)

# Write the output to the file
write_output_to_file(output)

# Run the query and get the result
query_result = run_query()
print(query_result)