---
layout: page
title: Graduates
permalink: /graduates/
---

# Graduates

These are the hardworking and dedicated students who have graduated from the course!

## Spring 2024

{% assign spring-2024-grads = site.graduates | where: 'quarter', 'spring-2024' %}
{% for graduate in spring-2024-grads %}
{{ graduate }}
{% endfor %}

## Autumn 2023

{% assign autumn-2023-grads = site.graduates | where: 'quarter', 'autumn-2023' %}
{% for graduate in autumn-2023-grads %}
{{ graduate }}
{% endfor %}

## Spring 2023

{% assign spring-2023-grads = site.graduates | where: 'quarter', 'spring-2023' %}
{% for graduate in spring-2023-grads %}
{{ graduate }}
{% endfor %}

## Winter 2023

{% assign winter-2023-grads = site.graduates | where: 'quarter', 'winter-2023' %}
{% for graduate in winter-2023-grads %}
{{ graduate }}
{% endfor %}
