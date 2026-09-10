/* WMRL project page: sticky nav, scroll reveal, chart tooltips.
   Everything here is progressive enhancement — the page reads fine without it. */

(function () {
  'use strict';

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- sticky nav appears once the hero is behind you ---------- */

  var nav = document.getElementById('nav');
  var header = document.querySelector('header');
  if (nav && header && 'IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      nav.classList.toggle('show', !entries[0].isIntersecting);
    }, { rootMargin: '-70px 0px 0px 0px' }).observe(header);
  }

  /* ---------- reveal on scroll ---------- */

  var targets = document.querySelectorAll('.reveal');
  if (reduced || !('IntersectionObserver' in window)) {
    targets.forEach(function (el) { el.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });
    targets.forEach(function (el) { io.observe(el); });
  }

  /* ---------- chart tooltips ---------- */

  var tip = document.getElementById('tip');
  if (!tip) return;

  function show(el, x, y) {
    tip.innerHTML =
      '<span class="t-name"><i style="background:' + el.dataset.color + '"></i>' +
      el.dataset.name + '</span><span class="t-val">' + el.dataset.val + '</span>';
    tip.style.left = x + 'px';
    tip.style.top = (y - 12) + 'px';
    tip.classList.add('on');
  }

  function hide() { tip.classList.remove('on'); }

  document.querySelectorAll('.chart').forEach(function (chart) {
    var hits = chart.querySelectorAll('.hit');
    if (!hits.length) return;

    hits.forEach(function (hit) {
      function enter(ev) {
        var r = hit.getBoundingClientRect();
        show(hit, r.left + r.width / 2, r.top);
        chart.classList.add('dim');
        // the hit rect trails the mark it belongs to, sometimes past its label
        var mark = hit.previousElementSibling;
        for (var i = 0; i < 4 && mark && !mark.classList.contains('mark'); i++) {
          mark = mark.previousElementSibling;
        }
        if (mark && mark.classList.contains('mark')) mark.classList.add('on');
        if (ev && ev.cancelable) ev.preventDefault();
      }
      function leave() {
        hide();
        chart.classList.remove('dim');
        chart.querySelectorAll('.mark.on').forEach(function (m) { m.classList.remove('on'); });
      }
      hit.addEventListener('mouseenter', enter);
      hit.addEventListener('mouseleave', leave);
      hit.addEventListener('touchstart', enter, { passive: false });
      hit.addEventListener('touchend', leave);
    });

    chart.addEventListener('mouseleave', function () {
      hide();
      chart.classList.remove('dim');
      chart.querySelectorAll('.mark.on').forEach(function (m) { m.classList.remove('on'); });
    });
  });

  window.addEventListener('scroll', hide, { passive: true });
})();
