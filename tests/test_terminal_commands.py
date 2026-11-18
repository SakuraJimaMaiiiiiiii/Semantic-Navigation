import unittest
from navigation.terminal_commands import parse_line, TerminalCommands


class TerminalTests(unittest.TestCase):
    def test_ordered_targets_and_commands(self):
        self.assertEqual(parse_line('vehicle_02')[0]['target']['instance_id'], 'vehicle_02')
        self.assertEqual([c['target']['instance_id'] for c in parse_line('vehicle_03 vehicle_01')],
                         ['vehicle_03', 'vehicle_01'])
        self.assertEqual(parse_line('返航'), [{'command':'return_home'}])
        self.assertEqual(parse_line('quit'), [{'command':'shutdown'}])
        self.assertEqual(parse_line(' '), [])
        self.assertEqual(parse_line('vehicle_02 home'), [
            {'command':'navigate','target':{'instance_id':'vehicle_02'}}, {'command':'return_home'}])
        with self.assertRaises(ValueError): parse_line('vehicle_02 home vehicle_01')

    def run_console(self, fail=False):
        class Server:
            def __init__(self): self.calls=[]
            def descriptor(self): return {}
            def dispatch(self, command):
                self.calls.append(command)
                return {'job_id':str(len(self.calls))}
            def status(self, identity):
                return {'job':{'id':identity, 'state':'failed' if fail else 'succeeded',
                               'result':{'success':not fail}}}
        inputs=iter(['vehicle_03 vehicle_01'])
        prompts=[]
        def read(prompt):
            prompts.append(prompt)
            try: return next(inputs)
            except StopIteration: raise EOFError
        server=Server()
        console=TerminalCommands(server,reader=read,writer=lambda x:None)
        console.run()
        return server.calls, prompts

    def test_sequential_completion(self):
        self.assertEqual([x['target']['instance_id'] for x in self.run_console()[0]],['vehicle_03','vehicle_01'])
        prompts=self.run_console()[1]
        self.assertEqual(len(prompts),2)
        self.assertTrue(all('请输入导航点' in prompt for prompt in prompts))

    def test_failure_cancels_remaining_targets(self):
        calls, prompts=self.run_console(fail=True)
        self.assertEqual(len(calls),1)
        self.assertTrue(all('请输入导航点' in prompt for prompt in prompts))

    def test_stop_during_input_cannot_dispatch(self):
        class Server:
            def dispatch(self, command): raise AssertionError('Must not dispatch after close')
        console=TerminalCommands(Server(),writer=lambda x:None)
        def read(prompt):
            console.close()
            return 'vehicle_02'
        console.reader=read
        console.run()


if __name__ == '__main__': unittest.main()
