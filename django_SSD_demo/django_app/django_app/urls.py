from django.conf.urls import url
from django.contrib import admin
from django.views.static import serve
from . import search

urlpatterns = [
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
    url(r'^search-traffic',search.search_traffic),
    url(r'^result_image/(?P<path>.*)',serve,{'document_root':'/usr/src/result_image'}),
]



