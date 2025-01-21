from batch_request import process_requests
from csv_manager import store_to_csv, read_from_csv
import asyncio


async def main():
    request_list = read_from_csv("queries.csv", "query")
    # request_list = ["hi", "how are you? "]
    response_texts = await process_requests(request_list)
    responses = []
    
    for request, response_text in zip(request_list, response_texts):
        responses.append({"request": request, "response": response_text})

    store_to_csv(responses, "responses.csv")
    
if __name__ == "__main__":
    asyncio.run(main())