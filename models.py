import datetime
import json
import os
import uuid
from typing import List

from gpt3_helpers import gpt3_embedding
from utils import timestamp_to_datetime


class Message:
    def __init__(self, author: str, text: str, timestamp_sent: str = None, vector: list = None, _uuid: str = None):
        if _uuid is None:
            _uuid = uuid.uuid4()
        if timestamp_sent is None:
            timestamp_sent = datetime.datetime.now().timestamp()
        if vector is None:
            vector = gpt3_embedding(text)
        self.author = author
        self.text = text

        self.timestamp = timestamp_sent
        self.vector = vector
        self.__uuid = _uuid

    def get_string(self):
        return self.__str__()

    def get_uuid(self):
        return self.__uuid

    def __dict__(self):
        return {'message': self.text, 'author': self.author, 'timestamp': self.timestamp, 'vector': self.vector, 'uuid': str(self.__uuid)}

    def __str__(self):
        return f'{self.author} at {timestamp_to_datetime(self.timestamp)}: {self.text}'

    @classmethod
    def from_dict(cls, msg):
        return cls(msg['author'], msg['message'], msg['timestamp'], msg['vector'], uuid.UUID(msg['uuid']))


class Note:
    def __init__(self, note_text, timestamp=None):
        self.note_text = note_text
        if timestamp is None:
            self.timestamp = datetime.datetime.now().timestamp()

    @classmethod
    def generate_note_from_conversation(cls, conversation: "Conversation"):
        message_list = conversation.get_messages()
        # TODO
        pass

    def __dict__(self):
        return {'note_text': self.note_text, 'timestamp': self.timestamp}

    def __str__(self):
        return f'Note: {self.note_text}'

    @classmethod
    def from_dict(cls, note_dict):
        return cls(note_dict['note_text'], note_dict['timestamp'])


class Conversation:
    def __init__(self, messages: List[Message] = None, notes: List[Note] = None):
        if messages is None:
            messages = []
        if notes is None:
            notes = []
        self.messages = messages
        self.notes = notes

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def compress_notes(self):
        pass

    def get_last_messages_in_string(self, limit: int):
        message_list = self.messages[-limit:]
        return '\n'.join([msg.get_string() for msg in message_list])

    def get_notes_as_string(self):
        return '\n- '.join([i.note_text for i in self.notes]) if self.notes else 'No notes on this conversation so far.'

    def get_notes(self):
        return self.notes

    def add_note(self, note: Note):
        self.notes.append(note)

    def set_notes(self, notes: list):
        self.notes = notes

    def __dict__(self):
        return {'messages': [msg.__dict__() for msg in self.messages], 'notes': [note.__dict__() for note in self.notes]}

    @classmethod
    def load(cls):
        try:
            with open('chat_logs/conversation.json', 'r', encoding='utf-8') as f:
                return Conversation().from_dict(json.load(f))
        except FileNotFoundError:
            return Conversation()

    def save(self):
        with open(f'chat_logs/conversation.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.__dict__(), outfile, ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, conversation_dict):
        messages = [Message.from_dict(msg) for msg in conversation_dict['messages']]
        notes = [Note.from_dict(note) for note in conversation_dict['notes']]
        return cls(messages, notes)
