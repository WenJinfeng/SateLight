
import time
import logging
# import subprocess 

#5AzXcqjKNZQXMu74yspREWjPM7SwEZUdFZ6mPdAvh9pKnXTrBzqGVaKfw2DN2aN1PPHt4Fz4jKTYiMwUp5NLW5JCoRDwB9WIo51kpEf30wBdK4x2HyvwGT9o34QJjjQ27fkpHbnZntUjrfDY2LYj2yBWptfCbTx4zXKUNPbQGLtb7c8rsIPHXReb6ixO9AmzAkMwK8t9PnsXPkuH7CBL5wWvl5HySeJh2tYr2S3qxTeEYoQJMKt5yip97DLwj5eM5kb8DZ5kZBpqjTu7v1YdYqbq7k9kEGuAgFcAaUtvdrtwf3aU6RjTdkQUtdG2xJxGHZFCpPY1CD8ebccn1vA3TzN9hRMUzvBz6RoNRn9SczSE6PAi3cdiYkqpdvXBcrULgYAF4GDUrGLXEfYvJzK9ReSH4JxBmiBRvELKwbFn5uEGmtkPN9oeKm5V9TYwpZNBXKhJcUnWaQzUuw2bD8kdoXhV7KBDZzZ4LbHpGR1TUSr5ZqqX9ewvRvd5FrJhkgI5AdZkvWq4vWgmhH9UvHTOtJ2TkueTuToYZwPkGyLtEsocOwFHLRgPSJvKBNx28mh9JdBZc6HXRD7ZrTuY2CuRPkpiRHmRiDNPhhTDaBh9j8rZjAGUXmsZUFxZgHGvWX28dn43J5UEHy0cbbDgGB1t7sow3BQPM2jXl9bVxCEHzWyudZNNwz2AtGFiJhBtjvW0d2Vyx5yi0l2qZyLR8AZy7MyYwGZRL3mPgDAgGC7W3Ktz7k0I9qLU8rJhJuAEnoXeTjY6gL4A8hjqmdDKvMQKRBIDxNk5OZ6TVWW9AnWjv9epRuvvKVA2cYhn0XUYzNU8IQmuPcJJc4oZ9R38VrkBXla3B1FVbMsACcrkFiT9yz2k7dN4ByMLNUzt5CgzY4Mwfxcd7MI1ltnrhq03mgGwMz2EN9i6XNmnhVEnjUpnXOkYTHGzdxBo3K2g6j70cyA9GmN3oGZQcrPlspQeW9peU15IMu0EjZDt9EmKRhtBGjlzEYaYt88pTH69Co7xky1R8DDfE7bnY2i80HLnmxivZRuX
import grpc  # type: ignore

import client_pb2 as client_pb2  # type: ignore
import client_pb2_grpc as client_pb2_grpc  # type: ignore

KEYGROUP = "simple"


# with open("cert.crt") as f:
# crt = f.read().encode()

# creds = grpc.ssl_channel_credentials(ca, key, crt)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")





# gateway = get_gateway_ip() 
# print(f"Gateway IP: {gateway}") 


MOCK_FRED_ADDRESS = 'localhost:9001'
logging.info(f"get_fn: Connecting to MockFReD at {MOCK_FRED_ADDRESS}")

channel = grpc.insecure_channel(MOCK_FRED_ADDRESS)
stub = client_pb2_grpc.ClientStub(channel)


def get(key: str) -> str:
    rr = client_pb2.ReadRequest(
        keygroup=KEYGROUP,
        id=key,
    )
    logging.debug(f"get_fn: Sending ReadRequest for key: {key}")
    try:
        response = stub.Read(rr)
        if response and response.data:
           
            value = response.data[0].val
            logging.debug(f"get_fn: Received value '{value}' for key '{key}'")
            return value
        else:
            logging.warning(f"get_fn: No data found for key '{key}' in response.")
            
            return "" 
    except grpc.RpcError as e:
        logging.error(f"get_fn: gRPC error getting value for key '{key}': {e.code()} - {e.details()}")
        
        raise 


def fn(inp: str) -> str: 
    # input is just a key
    t1 = time.perf_counter()
    logging.info(f"get_fn (FaaS entry): Received request to get key '{inp}'")

    val = "" 
    try:
        val = get(inp)
    except Exception as e: 
        logging.error(f"get_fn (FaaS entry): Error getting value for key '{inp}': {e}")
        
        raise 

    logging.info(f"get_fn (FaaS entry): Time taken: {time.perf_counter() - t1:.4f}s. Returning value for '{inp}'.")
    return val


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) 

    

    print("Testing get_fn locally...")
    try:
        

        retrieved_value_nonexistent = fn("nonexistentkey")
        print(f"Attempted to retrieve value for 'nonexistentkey': {retrieved_value_nonexistent} (expected error or empty)")

    except Exception as e:
        print(f"Error during local test of get_fn: {e}")
