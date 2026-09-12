"""Regression coverage without scheduling jobs or changing the desktop."""
import shlex
import unittest
from unittest.mock import patch

import executor


class SchedulingTests(unittest.TestCase):
    def calls(self):
        return [('timer.set', {'minutes': 1}),
                ('alarm.set', {'hour': 19}),
                ('reminder.set', {'minutes': 1, 'text': 'hello'})]

    def test_missing_scheduler_never_claims_success(self):
        with patch.object(executor, '_which_any', return_value=None):
            for tool, params in self.calls():
                self.assertFalse(executor.execute(tool, params)['ok'])

    def test_failed_scheduler_never_claims_success(self):
        with patch.object(executor, '_which_any', side_effect=lambda name: '/usr/bin/' + name), \
             patch.object(executor, '_run', return_value=(False, 'atd unavailable')):
            for tool, params in self.calls():
                result = executor.execute(tool, params)
                self.assertFalse(result['ok'])
                self.assertIn('atd unavailable', result['output'])

    def test_notification_text_remains_one_shell_argument(self):
        payload = "Sam's tea'; printf INJECTED; #\n$(whoami)"
        with patch.object(executor, '_which_any', side_effect=lambda name: '/usr/bin/' + name), \
             patch.object(executor, '_run', return_value=(True, 'queued')) as run:
            for tool, params in self.calls():
                params.update(label=payload, text=payload)
                self.assertTrue(executor.execute(tool, params)['ok'])
                args = shlex.split(run.call_args.kwargs['input'])
                self.assertEqual(args[:5], ['/usr/bin/notify-send', '-u', 'critical', '--',
                                          {'timer.set': 'Cozy timer', 'alarm.set': 'Cozy alarm',
                                           'reminder.set': 'Cozy reminder'}[tool]])
                self.assertEqual(len(args), 6)
                self.assertIn(payload, args[-1])

    def test_hindi_clock_periods(self):
        for phrase, expected in [('shaam 7 baje', (19, 0)), ('raat 9 baje', (21, 0)),
                                 ('saam 7 baje', (19, 0)), ('subah 7 baje', (7, 0)),
                                 ('subah 12 baje', (0, 0)), ('12pm', (12, 0))]:
            self.assertEqual(executor._parse_clock_time(phrase), expected)

    def test_alarm_uses_unambiguous_at_timestamp(self):
        with patch.object(executor, '_which_any', return_value='/usr/bin/at'), \
             patch.object(executor, '_run', return_value=(True, 'queued')) as run:
            executor.alarm_set({'hour': 19, 'minute': 30})
            self.assertEqual(run.call_args.args[0][1], '-t')
            self.assertRegex(run.call_args.args[0][2], r'^\d{12}$')


class CalculatorTests(unittest.TestCase):
    def test_normal_arithmetic(self):
        for expr, result in [('2 + 3 * 4', '14'), ('(2+3)**2', '25'),
                             ('-4 / 2', '-2'), ('7 % 3', '1'), ('9**0.5', '3')]:
            self.assertEqual(executor.calc_compute({'expression': expr}), (True, f'{expr} = {result}'))

    def test_expensive_and_non_numeric_expressions_are_rejected(self):
        for expr in ['9**9**9', '2**10000000', '1e999', 'True+1', '__import__("os")',
                     '(1).__class__', '1/0', '(-1)**0.5', '1+' * 150 + '1']:
            with self.subTest(expr=expr):
                self.assertFalse(executor.calc_compute({'expression': expr})[0])


class DesktopActionTests(unittest.TestCase):
    def test_missing_levels_never_silently_change_settings(self):
        self.assertFalse(executor.system_volume_set({})[0])
        self.assertFalse(executor.system_brightness_set({})[0])

    def test_screenshot_failure_is_not_wrapped_as_success(self):
        with patch.object(executor, '_which_any', side_effect=lambda name: '/usr/bin/grim' if name == 'grim' else None), \
             patch.object(executor, '_run', return_value=(False, 'capture failed')):
            result = executor.execute('screenshot.take', {})
        self.assertFalse(result['ok'])
        self.assertEqual(result['output'], 'capture failed')

    def test_missing_application_is_reported_instead_of_opened_as_a_file(self):
        with patch.object(executor, '_which_any', return_value=None):
            self.assertEqual(executor.app_open({'name': 'definitely-missing'}),
                             (False, 'application not found: definitely-missing'))


if __name__ == '__main__':
    unittest.main()
