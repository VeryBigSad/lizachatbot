import datetime
import json
from time import time
from uuid import uuid4

import openai

from constants import USERNAME, BOT_NAME
from gpt3_helpers import vector_similarity, gpt3_embedding, gpt3_completion
from models import Message, Conversation


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = vector_similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    notes.split('\n')
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector}
    filename = 'notes_%s.json' % time()
    save_json('notes/%s' % filename, info)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output


def get_user_input() -> Message:
    user_input = input(f'{USERNAME}: ')
    msg = Message(USERNAME, user_input)
    return msg


def main():
    conversation = Conversation.load_convo()
    while True:
        # step 1 - get input
        # step 2 - gather all information about the conversation (load memories, notes, wikipedia maybe?)
        # step 3 - use vector search to find information in our conversation/other information sources
        # step 4 - compile an answer using such information
        # step 5 - put out an answer
        # step 6 - create a memory of the conversation
        # step 7 - repeat

        # step 1
        user_input = input(f'{USERNAME}: ')
        message = get_user_input()
        conversation.add_message(message)

        # step 2
        gather_info_prompt = open_file('prompts/prompt_conversation_prepare_info.txt').replace('<<INPUT>>', message.text)
        queries = gpt3_completion(gather_info_prompt)
        conversation

        timestamp = time()

        timestring = timestamp_to_datetime(timestamp)
        message = f'{USERNAME}: {user_input}'

        info = {'speaker': 'USER', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()),
                'timestring': timestring}
        filename = 'log_%s_USER.json' % timestamp
        save_json('chat_logs/%s' % filename, info)
        #### load conversation
        conversation = load_convo()
        #### compose corpus (fetch memories, etc)
        if len(conversation) > 1:
            memories = fetch_memories(vector, conversation, 10)  # pull episodic memories
            # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
            notes = summarize_memories(memories)
            # TODO - search existing notes first
            recent = get_last_messages(conversation, 16)
        else:
            notes = ''
            recent = ''

        prompt = open_file('prompts/prompt_response.txt').replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent)
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s' % (BOT_NAME, output)
        info = {'speaker': BOT_NAME, 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()),
                'timestring': timestring}
        filename = f'log_{time()}_{BOT_NAME}.json'
        save_json('chat_logs/%s' % filename, info)
        #### print output
        print(f'\n\n{BOT_NAME}: {output}')


if __name__ == '__main__':
    openai.api_key = open_file('openaiapikey.txt')
    main()
