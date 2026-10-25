"""Argument Gym + legacy speaking practice, on the user's existing project.
Run: streamlit run app.py. Recording uses the mic on the computer running Python.
"""
import asyncio
import json
from datetime import datetime

import numpy as np
import sounddevice as sd
import streamlit as st

from coach_test import SAMPLE_RATE, CHUNK_MS
from dashboard import show_dashboard
from debate import compact_summary, final_recap, grade_answer, next_challenge
from history import delete_record, load_records, save_record
from metrics import Coach, chunk_db
from opponent import speak_opponent
from speak import make_record, record
from topics import CATEGORIES, pick_topic

st.set_page_config(page_title='Argument Gym', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden; height: 0;}
    .stApp {background: #0c0e12;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1480px;}
    [data-testid="stSidebar"] {background: #10131a; border-right: 1px solid #252a33;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: #9aa0ab;
    }
    .app-masthead {margin: 0 0 1.35rem; padding-bottom: 1rem; border-bottom: 1px solid #252a33;}
    .app-kicker {font-size: 0.7rem; letter-spacing: 0.22em; text-transform: uppercase; color: #8b8070; margin: 0 0 0.35rem;}
    .app-title {font-size: 1.85rem; font-weight: 600; letter-spacing: 0.04em; margin: 0; color: #f2efe8;}
    .app-lede {margin: 0.45rem 0 0; color: #9aa0ab; font-size: 0.95rem;}
    .panel-label {font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: #c4b49a; margin: 0 0 0.2rem;}
    .panel-title {font-size: 1.15rem; font-weight: 600; color: #f2efe8; margin: 0 0 0.35rem;}
    .panel-desc {color: #9aa0ab; font-size: 0.9rem; margin: 0 0 0.85rem;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #15181f;
        border: 1px solid #2a303a !important;
        padding: 0.35rem 0.15rem 0.2rem;
    }
    div[data-testid="stHorizontalBlock"] {gap: 1rem; align-items: stretch;}
    .stButton > button {border-radius: 2px; border: 1px solid #3a414d; background: #1c212b; color: #e4e2dc;}
    .stButton > button[kind="primary"] {background: #c4b49a; color: #14161c; border: 1px solid #c4b49a; font-weight: 600;}
    .stButton > button[kind="primary"]:hover {background: #d6c4a8; color: #14161c;}
    [data-testid="stExpander"] {background: #10131a; border: 1px solid #2a303a;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-masthead">
  <p class="app-kicker">Local debate training</p>
  <h1 class="app-title">Argument Gym</h1>
  <p class="app-lede">Spar with an AI opponent, then review delivery and argument quality in the same workspace.</p>
</div>
""", unsafe_allow_html=True)

user = st.sidebar.text_input('Local profile', 'me')
try:
    load_records(user)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


def panel_header(index, title, desc):
    st.markdown(f'<p class="panel-label">Panel {index:02d}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="panel-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="panel-desc">{desc}</p>', unsafe_allow_html=True)


def timer_widgets(prefix):
    """Display a circular countdown for each timed recording (debate or practice)."""
    placeholder = st.empty()
    previous_second = [None]

    def tick(remaining, total):
        # The recorder calls us about every 0.2 seconds; redraw only once a second.
        whole_seconds = max(0, min(int(total), int(remaining + 0.999)))
        if whole_seconds == previous_second[0]:
            return
        previous_second[0] = whole_seconds

        minutes, seconds = divmod(whole_seconds, 60)
        fraction = whole_seconds / max(total, 1)
        angle = max(0, min(360, round(360 * fraction, 2)))
        ring_color = '#c45c4a' if whole_seconds <= 10 else '#c4b49a'
        label = "TIME'S UP" if whole_seconds == 0 else 'TIME REMAINING'

        placeholder.markdown(
            f"""
            <div style="display:flex;justify-content:center;margin:16px 0 22px;">
              <div role="timer" aria-label="{minutes} minutes {seconds} seconds remaining"
                   style="width:210px;height:210px;border-radius:50%;padding:11px;
                          box-sizing:border-box;
                          background:conic-gradient({ring_color} {angle}deg,
                                                     rgba(128,128,128,.18) {angle}deg 360deg);">
                <div style="width:100%;height:100%;border-radius:50%;
                            background:#15181f;
                            display:flex;flex-direction:column;align-items:center;
                            justify-content:center;">
                  <div style="font-size:43px;font-weight:750;line-height:1.15;
                              font-variant-numeric:tabular-nums;
                              color:#f2efe8;">
                    {minutes:02d}:{seconds:02d}
                  </div>
                  <div style="font-size:11px;letter-spacing:1.5px;margin-top:9px;
                              font-weight:650;color:#9aa0ab;">
                    {label}
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return tick


def capture(seconds):
    coach = Coach(baseline_db=st.session_state.baseline, keep_text=True, track_voice=True)
    asyncio.run(record(coach, seconds, live=False, on_tick=timer_widgets('capture')))
    return coach


def transcript_notice():
    st.caption('Your microphone audio is sent to Cartesia for transcription. The opponent '
               'voice is also generated by Cartesia and plays through this computer\'s speakers. '
               'Transcribed answers go to Claude for feedback. By default, transcripts are not '
               'saved locally.')


def require_calibration(button_key):
    if 'baseline' in st.session_state:
        st.caption(f'Microphone baseline: {st.session_state.baseline:.1f} dBFS')
    if st.button('Calibrate microphone (5 seconds)', key=f'calibrate_microphone_{button_key}'):
        try:
            with st.spinner('Speak normally into the microphone for five seconds'):
                audio = sd.rec(5 * SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
                sd.wait()
            raw = audio.tobytes()
            size = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
            dbs = [chunk_db(raw[i:i + size]) for i in range(0, len(raw) - size + 1, size)]
            active = [v for v in dbs if v > max(dbs) - 20 and v > -65]
            if len(active) < 10:
                st.warning('Not enough speech detected. Check mic access and try again.')
            else:
                st.session_state.baseline = float(np.median(active))
                st.success('Microphone calibrated.')
        except Exception as exc:
            st.error(f'Microphone problem: {exc}')
    return 'baseline' in st.session_state


def start_debate(topic, side, rounds, seconds, save_text):
    st.session_state.duel = {
        'user': user, 'topic': topic, 'side': side, 'round_limit': rounds,
        'seconds': seconds, 'save_text': save_text, 'turns': [],
        'question': (f'You are arguing {side.lower()} this motion: {topic}. '
                     'Deliver your opening argument now.'),
        'challenge_focus': 'opening', 'used_questions': [], 'finished': False,
        'spoken_for': None,
    }


def finish_debate(duel):
    if duel['finished']:
        return
    turns = duel['turns']
    recap = final_recap(duel['topic'], turns)
    report = ('**Overall:** ' + recap['overall'] + '\n\n'
              + '**Progress:** ' + recap['biggest_improvement'] + '\n\n'
              + '**Next drill:** ' + recap['next_drill'] + '\n\n'
              + '**Next goal:** ' + recap['next_goal'])
    sanitized_turns = []
    for turn in turns:
        t = {k: turn[k] for k in ('round', 'question', 'focus', 'summary', 'evaluation')}
        if duel['save_text']:
            t['transcript'] = turn['transcript']
        sanitized_turns.append(t)
    entry = {'when': datetime.now().isoformat(timespec='seconds'), 'mode': 'debate',
             'category': 'Debate sparring', 'topic': duel['topic'], 'side': duel['side'],
             'target_s': duel['seconds'] * len(turns), 'summary': compact_summary(turns),
             'report': report, 'debate_turns': sanitized_turns, 'recap': recap}
    path = save_record(duel['user'], entry)
    duel['finished'] = True
    duel['saved_as'] = path.name
    duel['recap'] = recap


def deliver_opponent_line(duel, force=False):
    """Speak the current opponent turn once, using this computer's speakers."""
    line = (duel.get('question') or '').strip()
    if not line:
        return
    if not force and duel.get('spoken_for') == line:
        return
    try:
        with st.spinner('Opponent speaking...'):
            speak_opponent(line)
        duel['spoken_for'] = line
        duel.pop('voice_error', None)
    except Exception as exc:
        duel['spoken_for'] = line
        duel['voice_error'] = str(exc)


with st.sidebar:
    st.markdown('### Microphone')
    calibrated = require_calibration('workspace')
    st.markdown('---')
    transcript_notice()

sparring_col, coach_col = st.columns(2, gap='medium')

with sparring_col, st.container(border=True):
    panel_header(1, 'AI Sparring',
                 'Speak your case into the mic. The opponent answers out loud, then you respond.')
    duel = st.session_state.get('duel')
    if duel and duel['user'] != user:
        st.info('You changed profiles. Start a new debate for this profile.')
        duel = None

    if not duel or duel['finished']:
        topic_pool = CATEGORIES.get('Persuasion and debate', [])
        if not topic_pool:
            topic_pool = [t for seq in CATEGORIES.values() for t in seq]
        if 'duel_topic' not in st.session_state or st.session_state.duel_topic not in topic_pool:
            st.session_state.duel_topic = topic_pool[0]
        def generate_new_debate_topic():
            options = [topic for topic in topic_pool if topic != st.session_state.duel_topic]
            if options:
                import random
                st.session_state.duel_topic = random.choice(options)

        st.selectbox('Choose a debate topic', topic_pool, key='duel_topic')
        st.button(
            'Generate new debate topic',
            key='generate_new_debate_topic',
            on_click=generate_new_debate_topic,
        )
        side = st.radio('Which position will you defend?', ['For', 'Against'], horizontal=True)
        rounds = st.slider('Number of responses (opening + challenges)', 2, 5, 3)
        seconds = st.select_slider('Time per response', options=[30, 45, 60, 90, 120], value=60)
        save_text = st.checkbox('Save my response transcripts in History', key='debate_save', value=False)
        if st.button('Start debate', type='primary', disabled=not calibrated):
            start_debate(st.session_state.duel_topic, side, rounds, seconds, save_text)
            st.rerun()
        if not calibrated:
            st.info('Calibrate the microphone to enable recording.')

    if duel and not duel['finished']:
        st.divider()
        idx = len(duel['turns']) + 1
        st.markdown(f'**Round {idx} of {duel["round_limit"]}** · You argue **{duel["side"].lower()}**')
        deliver_opponent_line(duel)
        st.caption('Opponent')
        st.info(duel['question'])
        if duel.get('voice_error'):
            st.warning(f'The opponent could not be heard: {duel["voice_error"]}')
        if duel['challenge_focus'] != 'opening':
            st.caption(f'Challenge focus: {duel["challenge_focus"]}')
        st.caption(f'Listen, then record. You will have {duel["seconds"]} seconds.')
        if st.button('Replay opponent', key='replay_opponent'):
            deliver_opponent_line(duel, force=True)
        if st.button('Record my response', type='primary', disabled=not calibrated):
            try:
                sd.stop()
                coach = capture(duel['seconds'])
                transcript = coach.transcript_text().strip()
                summary = coach.summary()
                with st.spinner('Evaluating your argument and preparing the next challenge...'):
                    evaluation = grade_answer(duel['topic'], duel['question'], transcript, idx)
                    turn = {'round': idx, 'question': duel['question'],
                            'focus': duel['challenge_focus'], 'transcript': transcript,
                            'summary': summary, 'evaluation': evaluation}
                    duel['turns'].append(turn)
                    if idx >= duel['round_limit']:
                        finish_debate(duel)
                    else:
                        challenge = next_challenge(duel['topic'], duel['question'], transcript,
                                                   evaluation, duel['used_questions'])
                        duel['used_questions'].append(challenge['question'])
                        duel['question'] = challenge['question']
                        duel['challenge_focus'] = challenge['focus']
                st.rerun()
            except Exception as exc:
                st.error(f'Round could not be completed: {exc}. You can retry this round.')
        if st.button('Finish and save debate early'):
            if duel['turns']:
                with st.spinner('Saving your debate...'):
                    finish_debate(duel)
                st.rerun()
            else:
                st.warning('Record at least one response before saving.')
        if st.button('Discard this unsaved debate'):
            st.session_state.pop('duel', None)
            st.rerun()

    if duel:
        if duel.get('finished'):
            st.success(f'Debate saved: speeches/{user}/{duel["saved_as"]}')
            st.markdown('### Final coaching report')
            recap = duel['recap']
            st.write(recap['overall'])
            st.write('**Biggest improvement:**', recap['biggest_improvement'])
            st.write('**Next drill:**', recap['next_drill'])
            st.write('**Next goal:**', recap['next_goal'])
        if duel['turns']:
            st.markdown('### Round-by-round feedback')
            for t in duel['turns']:
                with st.expander(f'Round {t["round"]} · {t["question"]}', expanded=t is duel['turns'][-1]):
                    ev = t['evaluation']
                    st.write('**Strength:**', ev['strength'])
                    st.write('**Improve:**', ev['weakness'])
                    st.write('**Try next:**', ev['actionable_tip'])
                    if ev.get('scores'):
                        st.bar_chart(ev['scores'])
                    st.caption('AI rubric ratings (1–5) are subjective and may reflect transcription errors.')
                    st.write('**Delivery measurements:**')
                    st.json({k: t['summary'].get(k) for k in
                        ('pace_excluding_pauses_wpm', 'fillers_per_min',
                         'avg_volume_vs_baseline_db', 'pitch_variation_semitones')})
        if duel.get('finished') and st.button('Start another debate'):
            st.session_state.pop('duel', None)
            st.rerun()

with coach_col, st.container(border=True):
    panel_header(2, 'Speaking Coach',
                 'Random-topic speaking sessions with delivery analysis.')
    category = st.selectbox('Topic category', list(CATEGORIES))
    key = 'practice_topic_' + category
    if key not in st.session_state:
        st.session_state[key] = pick_topic(category)
    if st.button('Another practice topic'):
        st.session_state[key] = pick_topic(category, avoid=st.session_state[key])
    topic = st.session_state[key]
    st.info(topic)
    seconds = st.select_slider('Speaking time', options=[30, 60, 90, 120], value=60, key='practice_duration')
    save_text = st.checkbox('Save transcript locally', value=False, key='practice_save')
    if st.button('Record practice speech', type='primary', disabled=not calibrated):
        try:
            coach = capture(seconds)
            with st.spinner('Preparing your speech report...'):
                entry, path = make_record(user, topic, category, seconds, coach, save_text)
            st.session_state.practice_result = (user, entry, path.name)
            st.rerun()
        except Exception as exc:
            st.error(f'Recording failed: {exc}')
    result = st.session_state.get('practice_result')
    if result and result[0] == user:
        st.success(f'Practice saved: speeches/{user}/{result[2]}')
        st.markdown(result[1]['report'])

st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
progress_col, history_col = st.columns(2, gap='medium')

with progress_col, st.container(border=True):
    panel_header(3, 'Progress',
                 'Delivery metrics and argument rubric across saved sessions.')
    records = load_records(user)
    show_dashboard(records)
    debates = [r for r in records if r.get('mode') == 'debate']
    if debates:
        st.markdown('**Argument rubric over time**')
        import pandas as pd
        rows = []
        for session in debates:
            ratings = [t.get('evaluation', {}).get('scores', {}) for t in session.get('debate_turns', [])]
            row = {'Date': session.get('when')}
            for dimension in ('claim', 'reasoning', 'evidence', 'responsiveness', 'organization'):
                vals = [r[dimension] for r in ratings if dimension in r]
                row[dimension] = sum(vals) / len(vals) if vals else None
            rows.append(row)
        chart = pd.DataFrame(rows)
        chart['Date'] = pd.to_datetime(chart['Date'], errors='coerce')
        chart = chart.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        st.line_chart(chart)
        st.caption('AI rubric ratings are subjective, not standardized assessment scores.')

with history_col, st.container(border=True):
    panel_header(4, 'History',
                 'Saved debates and practice sessions for this profile.')
    records = load_records(user)
    if not records:
        st.info('No sessions saved yet. Complete a practice session or a debate.')
    for item in reversed(records):
        name = item['_filename']
        kind = 'Debate' if item.get('mode') == 'debate' else 'Practice'
        with st.expander(f'{kind} · {item.get("when", "")} · {item.get("topic", "")}'):
            st.markdown(item.get('report', 'No report.'))
            st.json(item.get('summary', {}))
            if item.get('transcript'):
                st.text_area('Transcript', item['transcript'], disabled=True, key='t_' + name)
            for turn in item.get('debate_turns', []):
                st.markdown(f'**Round {turn["round"]}: {turn["question"]}**')
                st.json(turn.get('evaluation', {}))
                if turn.get('transcript'):
                    st.text_area('Answer', turn['transcript'], disabled=True,
                                 key=f'ans_{name}_{turn["round"]}')
            st.download_button('Download session JSON',
                json.dumps({k: v for k, v in item.items() if k != '_filename'}, indent=2),
                file_name=name, mime='application/json', key='download_' + name)
            if st.button('Delete session', key='delete_' + name):
                delete_record(user, name)
                st.rerun()
