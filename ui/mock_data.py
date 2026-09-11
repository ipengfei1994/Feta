# -*- coding: utf-8 -*-
"""
UI 演示语料：模拟学生社交媒体动态（英文，与主链路英文语料对齐）。

仅用于 UI 展示层：
  - 文本分布覆盖四档风险（Normal / Anxiety / Depression / Suicidal）与 8 大压力源，
    使 classifier 的模型判定与风险分布图有真实感数据可用。
  - 时间戳相对服务器启动时刻生成，保证「最近 24 小时 / 7 天 / 30 天」三个时间范围都有数据。
  - Suicidal 语料为演示用模拟文本（非真实用户数据），仅用于验证危机卡 / 转介链路。
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

# (作者名, handle) 池 —— 头像由前端按 handle 哈希生成，无需图片资源
AUTHORS = [
    ("Emily Chen", "emily_c"), ("Jake Morrison", "jake_m23"), ("Sofia Ramirez", "sofiar"),
    ("Liam O'Brien", "liamob"), ("Aisha Khan", "aisha.k"), ("Noah Williams", "noahw"),
    ("Mia Thompson", "mia_t"), ("Ethan Davis", "ethan_d"), ("Chloe Zhang", "chloe.z"),
    ("Daniel Park", "danielp"), ("Grace Liu", "gracel"), ("Marcus Johnson", "marcusj"),
    ("Olivia Brown", "oliviab"), ("Ryan Kim", "ryank"), ("Hannah Lee", "hannahlee"),
    ("Tyler Moore", "tylermo"), ("Zoe Adams", "zoe.a"), ("Kevin Wang", "kevinw"),
]

# 配图占位：(emoji, 描述) —— 前端用渐变块渲染，无外部图片依赖
POST_IMAGES = [
    ("📚", "desk with books"), ("🌙", "night sky"), ("☕", "coffee cup"),
    ("🏃", "morning run"), ("🌧️", "rainy window"), ("📝", "notes"),
]

# (文本, 相对小时偏移上限) —— 偏移在 [0, cap] 内随机，cap 越小越「新」
TEXT_POOL = [
    # ---- Normal（日常正向 / 中性）----
    ("Finally finished my group project presentation and it went really well!", 20),
    ("Morning run by the lake, then pancakes. Weekends are for recharging.", 30),
    ("Our study group actually understands statistics now. Small victories.", 40),
    ("Started a new podcast on my commute, huge quality of life upgrade.", 60),
    ("Made dinner with my roommate tonight, we laughed so much.", 26),
    ("Library session done. Three chapters, one coffee, zero regrets.", 18),
    ("Joined the photography club this week, best decision this semester.", 90),
    ("My plants are thriving and honestly so am I.", 150),
    ("Film night with the crew. We watched a terrible movie on purpose.", 34),
    ("Volunteered at the shelter today. Feeling grateful.", 200),
    ("Midterm schedule is out, planned my revision calendar already.", 46),
    ("New playlist for late night studying is unmatched.", 70),
    # ---- Academic_Stress（多为 Anxiety）----
    ("I have three exams in four days and I cannot retain anything I read.", 10),
    ("My GPA is slipping and every time I open my textbook I just freeze.", 14),
    ("I failed two quizzes this month. I study all night and still blank out in the exam hall.", 8),
    ("Deadline for the thesis proposal is Friday and I have not written a single word.", 6),
    ("Everyone in my class seems ahead of me. I am terrified of being the worst one.", 12),
    ("I cannot sleep because I keep replaying the exam questions in my head.", 5),
    ("Professor said my draft was disappointing in front of everyone. I feel like quitting.", 30),
    ("So much coursework piling up, I do not even know where to start anymore.", 22),
    # ---- Interpersonal / Family ----
    ("My roommate and I had a huge fight and now the dorm feels hostile.", 16),
    ("I feel invisible in my friend group. Nobody replies to my messages anymore.", 28),
    ("My parents keep comparing me to my cousin and I am so tired of it.", 40),
    ("Group project members are not responding and I am doing all the work alone.", 20),
    ("I had a panic attack before the presentation and had to leave the room.", 9),
    # ---- Career_Future ----
    ("Rejected from my third internship this month. Starting to think I am not good enough.", 24),
    ("Everyone around me has offers and I cannot even get an interview.", 48),
    ("I have no idea what I want to do after graduation and it keeps me up at night.", 36),
    # ---- Appearance_Anxiety ----
    ("I keep deleting photos of myself because I hate how I look in all of them.", 32),
    ("Someone commented on my weight today and I have not eaten properly since.", 15),
    # ---- Depression ----
    ("I have felt empty for weeks. Nothing is fun anymore, I just go through the motions.", 12),
    ("I skip classes now because getting out of bed feels impossible.", 8),
    ("Told my counselor I have been feeling numb. She suggested I keep a mood journal.", 60),
    ("I lie awake at night feeling like a burden to everyone around me.", 6),
    ("Nothing I do seems to matter. My grades, my friendships, all of it feels pointless.", 10),
    # ---- Suicidal（演示转介链路，均为模拟文本）----
    ("I want to disappear. Nobody would even notice I am gone.", 4),
    ("I keep thinking about hurting myself. I do not see a way out of this.", 3),
    ("Sometimes I think everyone would be better off without me.", 5),
    ("I wrote goodbye letters last night. I am so tired of fighting this.", 2),
]

SLEEP_TEXTS = [
    ("It is 4am again and I cannot fall asleep before my 8am lecture.", 7),
    ("Nightmares keep waking me up. I am exhausted all day.", 11),
]


def build_posts(now: datetime | None = None, seed: int = 42) -> list[dict]:
    """生成演示动态列表：按时间倒序，含作者/配图/互动计数/时间戳。"""
    rng = random.Random(seed)
    now = now or datetime.now()
    posts = []
    pid = 1000
    for text, cap_hours in TEXT_POOL + SLEEP_TEXTS:
        author = rng.choice(AUTHORS)
        hours_ago = rng.uniform(0.5, cap_hours)
        ts = now - timedelta(hours=hours_ago)
        has_img = rng.random() < 0.22
        emoji, desc = rng.choice(POST_IMAGES)
        posts.append({
            "id": f"p{pid}",
            "author": author[0],
            "handle": author[1],
            "text": text,
            "created_at": ts.isoformat(timespec="seconds"),
            "likes": rng.randint(0, 180),
            "comments": rng.randint(0, 24),
            "shares": rng.randint(0, 30),
            "image": {"emoji": emoji, "desc": desc,
                      "hue": rng.randint(180, 320)} if has_img else None,
        })
        pid += 1
    posts.sort(key=lambda p: p["created_at"], reverse=True)
    return posts


# 模拟评论（互动区演示用）
MOCK_COMMENTS = [
    ("mia_t", "Sending you a hug. Want to grab coffee and talk?"),
    ("danielp", "I felt exactly the same last semester. It does get better."),
    ("gracel", "The counseling center on campus is really kind, I have been there."),
    ("noahw", "You are not alone in this. DM me anytime."),
    ("emily_c", "Same here... glad someone said it out loud."),
    ("kevinw", "Break it into small steps. One task at a time, seriously."),
]
