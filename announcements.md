---
layout: page
title: Announcements
description: A feed containing all of the class announcements.
nav_order: 1
---

# Announcements

{% assign announcements = site.announcements | reverse %}
{% for announcement in announcements %}
{{ announcement }}
{% endfor %}
