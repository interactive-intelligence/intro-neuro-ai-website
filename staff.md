---
layout: page
title: Course Staff
description: A listing of all the course creators.
---

# Current Course Staff

Here are the awesome people who made this course possible. Feel free to contact us if you have any issues!

## Education Leads

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

# Previous Course Staff

## The Creators

{% assign creators = site.staffers | where: 'role', 'Creator' %}
{% for staffer in creators %}
{{ staffer }}
{% endfor %}

## Previous Education Leads

{% assign instructors = site.staffers | where: 'role', 'OldInstructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

## Autumn 2024 Teaching Assistants

{% assign teaching_assistants = site.staffers | where: 'role', '24au Teaching Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}

## Spring 2024 Teaching Assistants

{% assign teaching_assistants = site.staffers | where: 'role', '24sp Teaching Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}

## Autumn 2023 Teaching Assistants

{% assign teaching_assistants = site.staffers | where: 'role', '23au Teaching Assistant' %}
{% for staffer in teaching_assistants %}
{{ staffer }}
{% endfor %}
