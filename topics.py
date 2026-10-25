import random

CATEGORIES = {
    'Global events': [
        'How is artificial intelligence changing education?',
        'Should cities prioritize public transportation over parking?',
        'What is one global problem that needs more attention?',
        'Should social media platforms be responsible for misinformation?',
        'Is space exploration worth its cost?',
        'How can countries cooperate on climate change?',
    ],
    'Your emotions': [
        'Describe a time you felt proud and what you learned.',
        'Talk about something that makes you nervous.',
        'When did you last feel grateful and why?',
        'Explain how you handle disappointment.',
        'Describe a time you overcame a fear.',
        'What does confidence mean to you?',
    ],
    'Persuasion and debate': [
        'Argue for or against banning homework.',
        'Should voting be mandatory?',
        'Should students be allowed phones in class?',
        'Persuade someone to learn a second language.',
        'Should cities make downtown areas car-free?',
        'Is teamwork more important than individual talent?',
    ],
    'Storytelling': [
        'Tell the story of your most memorable day.',
        'Describe a place that feels like home.',
        'Tell a story about an unexpected challenge.',
        'Describe a person who changed your perspective.',
        'Tell the story of something you built.',
        'Describe a risk you took and what happened.',
    ],
    'Quick thinking': [
        'Sell me a pen.',
        'Invent a new holiday and explain it.',
        'Explain a difficult concept to a ten-year-old.',
        'Convince a friend to try a new food.',
        'What would you do with an extra hour each day?',
        'Describe an invention the world needs.',
    ],
}


def pick_topic(category, avoid=None):
    choices = [t for t in CATEGORIES[category] if t != avoid] or CATEGORIES[category]
    return random.choice(choices)
