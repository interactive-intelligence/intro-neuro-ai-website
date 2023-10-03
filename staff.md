---
layout: page
title: Course Staff
description: A listing of all the course creators.
---

# Course Staff

Here are the awesome people who made the course + website. Feel free to contact us if you have any issues!

## Education Lead

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

## Teaching Assistants

{% assign teaching_assistants = site.staffers | where: 'role', 'Teaching Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}

## The Creators

{% assign creators = site.staffers | where: 'role', 'Creator' %}
{% for staffer in creators %}
{{ staffer }}
{% endfor %}
