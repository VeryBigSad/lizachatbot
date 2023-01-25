from time import time
from typing import List
from uuid import uuid4

import openai

from constants import USERNAME, BOT_NAME
from gpt3_helpers import vector_similarity, gpt3_embedding, gpt3_completion
from models import Message, Conversation, Note
from utils import open_file, save_json, timestamp_to_datetime


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
    return Message(USERNAME, user_input)


def search_conversation(conversation: Conversation, message: Message) -> List[Message]:
    """
    Search the conversation for messages that are related to the given message

    :param conversation: Conversation object, the conversation to search
    :param message: Message object, that we want to find related messages for
    :return: List of messages that are related to the given message
    """
    message_list = conversation.get_messages()
    query_vector = gpt3_embedding(message.text)
    similarities = {}
    for message in message_list:
        # TODO: don't store the entire message in memory, just the vector/uuid
        similarities[message] = vector_similarity(query_vector, message.vector)
        # get the top 3 most similar messages
    ordered = [i[0] for i in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:6]]
    return ordered


def summarize_notes(notes):
    prompt = open_file('prompts/compress_notes.txt').replace('<<NOTES>>', '\n- '.join([note.note_text for note in notes]))
    result = gpt3_completion(prompt)
    return [Note(i.strip()) for i in result.split('- ')]


def main():
    conversation = Conversation.load()
    while True:

        # step 1 - get input
        # step 2 - gather all information about the conversation (load memories, notes, wikipedia maybe?)
        # step 3 - generate search queries to search info about the input
        # step 4 - use vector search to find information in our conversation/other information sources
        # step 5 - compile an answer using such information
        # step 6 - put out an answer
        # step 7 - create a memory of the conversation
        # step 8 - repeat

        # step 1
        message = get_user_input()
        conversation.add_message(message)

        # step 2
        last_6_messages = conversation.get_last_messages_in_string(12)  # get last 6 messages as a string
        notes = conversation.get_notes_as_string()  # get notes from the conversation
        # gather_info_prompt = (
        #     open_file('prompts/prompt_conversation_prepare_info.txt')
        #     .replace('<<CONVERSATION>>', last_6_messages)
        #     .replace('<<NOTES>>', notes)
        # )
        # step 3
        # search_queries = [i.strip() for i in gpt3_completion(gather_info_prompt).split('- ') if i.strip() != '']
        # step 4
        related_messages = search_conversation(conversation, message)
        # facts = [f"Question: {i}; Answer: {input(i)}" for i in search_queries] # lmao
        # step 5
        if len(related_messages) > 10:
            answer_prompt = (
                open_file('prompts/prompt_response.txt')
                .replace('<<CONVERSATION>>', last_6_messages)
                .replace('<<NOTES>>', notes)
                .replace('<<MESSAGES_RELATED>>', '\n'.join([i.get_string() for i in related_messages]))
                # .replace('<<FACTS>>', '\n'.join(facts))
            )
        else:
            answer_prompt = (
                open_file('prompts/prompt_response_in_new_conversation.txt').replace('<<CONVERSATION>>', last_6_messages)
            )
        # step 6
        answer = gpt3_completion(answer_prompt)
        print(f'{BOT_NAME}: {answer}')
        conversation.add_message(Message(BOT_NAME, answer))
        # step 7
        notes_prompt = open_file('prompts/prompt_notes.txt').replace('<<INPUT>>', last_6_messages)
        notes = [i.strip() for i in gpt3_completion(notes_prompt).split('- ')]

        [conversation.add_note(Note(note)) for note in notes]
        if len(conversation.get_notes()) > 10:
            # compress notes
            notes = conversation.get_notes()
            notes = summarize_notes(notes)
            conversation.set_notes(notes)
        # step 8
        conversation.save()


if __name__ == '__main__':
    openai.api_key = open_file('openaiapikey.txt')
    main()
