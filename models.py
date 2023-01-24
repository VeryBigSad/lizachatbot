import datetime
import json
import os

from gpt3_helpers import gpt3_embedding


class Message:
    def __init__(self, text: str, author: str, datetime_sent: str = None, vector: list = None):
        self.text = text
        self.author = author
        if datetime_sent is None:
            self.datetime = datetime.datetime.now()
        if vector is None:
            self.vector = gpt3_embedding(text)

    def get_string(self):
        return self.__str__()

    def __dict__(self):
        return {'message': self.text, 'author': self.author, 'datetime': self.datetime, 'vector': self.vector}

    def __str__(self):
        return f'{self.author} at {self.datetime}: {self.text}'

    @classmethod
    def from_dict(cls, msg):
        return cls(msg['message'], msg['author'])


class Note:
    def __init__(self, note_text):
        self.note_text = note_text
        self.datetime = datetime.datetime.now()

    @classmethod
    def generate_note_from_conversation(cls, conversation: "Conversation"):
        message_list = conversation.get_messages()
        # TODO
        pass

    def __dict__(self):
        return {'note_text': self.note_text, 'datetime': self.datetime}

    def __str__(self):
        return f'Note at {self.datetime}: {self.note_text}'


class Conversation:
    def __init__(self, messages: list = None, notes: list = None):
        if messages is None:
            self.messages = []
        if notes is None:
            self.notes = []

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def get_notes(self):
        return self.notes

    def add_note(self, note: Message):
        self.notes.append(note)

    def set_notes(self, notes: list):
        self.notes = notes

    @classmethod
    def load_convo(cls):
        files = os.listdir('chat_logs')
        files = [i for i in files if '.json' in i]  # filter out any non-JSON files
        result = list()
        for file in files:
            with open(f'chat_logs/{file}', 'r', encoding='utf-8') as infile:
                data = json.load(infile)
            result.append(data)
        ordered = sorted(result, key=lambda d: d['datetime'], reverse=False)  # sort them all chronologically
        messages = [Message.from_dict(msg) for msg in ordered]
        return cls(messages)


