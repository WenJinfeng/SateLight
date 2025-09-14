
import time
import json
import logging
# import subprocess 

import grpc  # type: ignore
import client_pb2 as client_pb2  # type: ignore
import client_pb2_grpc as client_pb2_grpc  # type: ignore
#5AzXcqjKNZQXMu74yspREWjPM7SwEZUdFZ6mPdAvh9pKnXTrBzqGVaKfw2DN2aN1PPHt4Fz4jKTYiMwUp5NLW5JCoRDwB9WIo51kpEf30wBdK4x2HyvwGT9o34QJjjQ27fkpHbnZntUjrfDY2LYj2yBWptfCbTx4zXKUNPbQGLtb7c8rsIPHXReb6ixO9AmzAkMwK8t9PnsXPkuH7CBL5wWvl5HySeJh2tYr2S3qxTeEYoQJMKt5yip97DLwj5eM5kb8DZ5kZBpqjTu7v1YdYqbq7k9kEGuAgFcAaUtvdrtwf3aU6RjTdkQUtdG2xJxGHZFCpPY1CD8ebccn1vA3TzN9hRMUzvBz6RoNRn9SczSE6PAi3cdiYkqpdvXBcrULgYAF4GDUrGLXEfYvJzK9ReSH4JxBmiBRvELKwbFn5uEGmtkPN9oeKm5V9TYwpZNBXKhJcUnWaQzUuw2bD8kdoXhV7KBDZzZ4LbHpGR1TUSr5ZqqX9ewvRvd5FrJhkgI5AdZkvWq4vWgmhH9UvHTOtJ2TkueTuToYZwPkGyLtEsocOwFHLRgPSJvKBNx28mh9JdBZc6HXRD7ZrTuY2CuRPkpiRHmRiDNPhhTDaBh9j8rZjAGUXmsZUFxZgHGvWX28dn43J5UEHy0cbbDgGB1t7sow3BQPM2jXl9bVxCEHzWyudZNNwz2AtGFiJhBtjvW0d2Vyx5yi0l2qZyLR8AZy7MyYwGZRL3mPgDAgGC7W3Ktz7k0I9qLU8rJhJuAEnoXeTjY6gL4A8hjqmdDKvMQKRBIDxNk5OZ6TVWW9AnWjv9epRuvvKVA2cYhn0XUYzNU8IQmuPcJJc4oZ9R38VrkBXla3B1FVbMsACcrkFiT9yz2k7dN4ByMLNUzt5CgzY4Mwfxcd7MI1ltnrhq03mgGwMz2EN9i6XNmnhVEnjUpnXOkYTHGzdxBo3K2g6j70cyA9GmN3oGZQcrPlspQeW9peU15IMu0EjZDt9EmKRhtBGjlzEYaYt88pTH69Co7xky1R8DDfE7bnY2i80HLnmxivZRuXGmtkPN9oeKm5V9TYwpZNBXKhJcUnWaQzUuw2bD8kdoXhV7KBDZzZ4LbHpGR1TUSr5ZqqX9ewvRvd5FrJhkgI5AdZkvWq4vWgmhH9UvHTOtJ2TkueTuToYZwPkGyLtEsocOwFHLRgPSJvKBNx28mh9JdBZc6HXRD7ZrTuY2CuRPkpiRHmRiDNPhhTDaBh9j8rZjAGUXmsZUFxZgHGvWX28dn43J5UEHy0cbbDgGB1t7sow3BQPM2jXl9bVxCEHzWyudZNNwz2AtGFiJhBtjvW0d2Vyx5yi0l2qZyLR8AZy7MyYwGZRL3mPgDAgGC7W3Ktz7k0I9qLU8rJhJuAEnoXeTjY6gL4A8hjqmdDKvMQKRBIDxNk5OZ6TVWW9AnWjv9epRuvvKVA2cYhn0XUYzNU8IQmbNsZ6MWSHcrXJ0R9DG6Rx5oLkNno7Zme4vDJohGVGyIjY2giVZySl0H2qR1drRw9bCRp6TOzUJNFq9RAXdILyDt4hQuUorvq1Q5ZwAK8bHxgOVxH3f4V1nEj4xErBod6DOVJ8cU2GbqO0ERihDRNjAOC9rkmDtgVx4gyWvczqEypzpIzkY0xZtShNNYhD6Z6PSJkNY2L3TUsFZTijCxq5vBXuXesvF4cBtNKJNkJYbQb5C9S58oxvwZ8AD5vPxnkARQtc1hYx9TR4UnRJISnhdBFSzMLZbgyjMIxjcG8ogj8WXJVVRf81cHvxpkpkmNHeP4yz2CUeZn15MYUnX2jqmrlMoxh1crlYv2EV1BpIOPqfya2uHvlIMgzmmiwvbXOVWZevZhgeXFPWNhnlgZ9TzuPcJJc4oZ9R38VrkBXla3B1FVbMsACcrkFiT9yz2k7dN4ByMLNUzt5CgzY4Mwfxcd7MI1ltnrhq03mgGwMz2EN9i6XNmnhVEnjUpnXOkYTHGzdxBo3K2g6j70cyA9GmN3oGZQcrPlspQeW9peU15IMu0EjZDt9EmKRhtBGjlzEYaYt88pTH69Co7xky1R8DDfE7bnY2i80HLnmxivZRuXqypozg58qDCqvQnMF6F3pMI0zyll4XZECaPlHLWchjFEFUSzA2jvUq1Yrk7VJpzrNkAwqZV3GQkbgdYgOkXBmA6wTNDNqIRJDW1LCVV1GHEiUG7AhGCk3IjTL4iLtxWD1NDujMc2yd2eApcA8cXkft3e48XLfHOlnAdwmtkRU91zIVH9mycFcQAzEQ4Q7GVtiDOcT4AAn07qIR6UBDZ4Qv4HPDe4SBAEi3hO

KEYGROUP = "simple"



logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")



MOCK_FRED_ADDRESS = 'localhost:9001' 
logging.info(f"put_fn: Connecting to MockFReD at {MOCK_FRED_ADDRESS}")

channel = grpc.insecure_channel(MOCK_FRED_ADDRESS)
stub = client_pb2_grpc.ClientStub(channel)


def put(key: str, value: str) -> None:
    
    
    ur = client_pb2.UpdateRequest(
        keygroup=KEYGROUP,
        id=key,
        data=str(value), 
    )
    logging.debug(f"put_fn: Sending UpdateRequest for key: {key}, value: {value}")
    try:
        response = stub.Update(ur) 
        logging.debug(f"put_fn: Update successful for key '{key}'. Response version: {response.version.version if response and response.version else 'N/A'}")
    except grpc.RpcError as e:
        logging.error(f"put_fn: gRPC error putting value for key '{key}': {e.code()} - {e.details()}")
        raise 


def fn(inp: str) -> str: 
    # input is a little json string
    # {"key": "sensor1", "value": "8.9"} 

    t1 = time.perf_counter()
    logging.info(f"put_fn (FaaS entry): Received request with payload: {inp}")

    try:
        parsed_input = json.loads(inp)
        key = parsed_input["key"]
        
        value = str(parsed_input["value"]) 
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"put_fn (FaaS entry): Invalid JSON input or missing key/value: {inp} - Error: {e}")
        
        raise ValueError(f"Invalid input: {inp}. Expected JSON with 'key' and 'value'.")


    logging.debug(f"put_fn (FaaS entry): Received value '{value}' for key '{key}'")

    try:
        put(key, value)
    except Exception as e: 
        logging.error(f"put_fn (FaaS entry): Error putting value for key '{key}': {e}")
        raise 

    time_taken = time.perf_counter() - t1
    logging.info(f"put_fn (FaaS entry): Time taken: {time_taken:.4f}s. Successfully processed put for key '{key}'.")
    return "OK"



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO) 

    

    print("Testing put_fn locally...")
    test_payload_str = '{"key": "testkey_put_local", "value": "local_put_value_123"}'
    print(f"Sending payload: {test_payload_str}")
    try:
        result = fn(test_payload_str)
        print(f"Result from put_fn: {result}")
        
       

    except Exception as e:
        print(f"Error during local test of put_fn: {e}")
