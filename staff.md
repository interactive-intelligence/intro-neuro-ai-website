---
layout: page
title: Creators
description: A listing of all the course creators.
---

# Creators

Here are the awesome people who made the course + website. Feel free to contact us if you have any issues!

## Education Lead

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

{% assign teaching_assistants = site.staffers | where: 'role', 'Teaching Assistant' %}
{% assign num_teaching_assistants = teaching_assistants | size %}
{% if num_teaching_assistants != 0 %}

## Other Creators

{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
{% endif %}
