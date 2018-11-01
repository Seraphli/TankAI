from concurrency import ConPool
from waldorf.client import WaldorfClient
from waldorf.cfg import WaldorfCfg
from waldorf.env import MajorCmd, MinorCmd, CmdPair, \
    Suite, SetupSuite
from util import get_path

import tank


class WaldorfPool(ConPool):
    def setup(self):
        cfg = WaldorfCfg(master_ip='192.168.5.190')
        cfg.debug = 0
        cfg.env_cfg.already_exist = 'remove'
        cfg.env_cfg.version_mismatch = 'remove'
        cfg.env_cfg.git_credential = open(get_path('.', _file=__file__)
                                          + '/credential', 'rb').read()
        cfg.env_cfg.default_timeout = 310
        cfg.result_timeout = 10
        cfg.retry_times = 5
        self.waldorf_client = WaldorfClient(cfg, limit=self.limit)

        pairs = [
            CmdPair(MajorCmd.CREATE_ENV,
                    args=['$HOME/Python/3.6.5/bin/python3']),
            CmdPair(MajorCmd.CHECK_PY_VER, pattern='3.6.5')
        ]

        suites = [
            SetupSuite(
                Suite([CmdPair(MinorCmd.CREATE_SESS),
                       CmdPair(MinorCmd.SOURCE_ENV),
                       CmdPair(MinorCmd.RUN_CMD, args=['python', '>>>'])],
                      [CmdPair(MinorCmd.RUN_CMD, pattern='No module',
                               exist=False,
                               args=['import tank', '>>>']),
                       CmdPair(MinorCmd.RUN_CMD,
                               args=['tank.__version__', '>>>'],
                               pattern=tank.__version__)],
                      [CmdPair(MinorCmd.RUN_CMD, args=['exit()']),
                       CmdPair(MinorCmd.CLOSE_SESS)]),
                Suite([CmdPair(MinorCmd.CREATE_SESS),
                       CmdPair(MinorCmd.SOURCE_ENV),
                       CmdPair(MinorCmd.RUN_CMD, args=['cd'])],
                      [CmdPair(MinorCmd.GIT_CLONE, args=[
                          'git clone '
                          'http://server.levelup.io/liyue/tank.git',
                          'http://server.levelup.io/liyue/tank.git']),
                       CmdPair(MinorCmd.RUN_CMD, args=['cd tank']),
                       CmdPair(MinorCmd.RUN_CMD, args=['pip install -U .']),
                       CmdPair(MinorCmd.RUN_CMD, args=['cd ..']),
                       CmdPair(MinorCmd.RUN_CMD, args=['rm -rf tank'])],
                      [CmdPair(MinorCmd.CLOSE_SESS)])
            )
        ]
        resp = self.waldorf_client.get_env(self.name, pairs, suites)
        for hostname, r in resp:
            if r[0] < 0:
                raise Exception(hostname, r[1])

    def reg_task(self, tasks):
        for task in tasks:
            self.waldorf_client.reg_task(task)
        self.waldorf_client.freeze()

    def apply(self, func, args, callback):
        self.waldorf_client.test_submit(func, args, callback)

    def apply_async(self, func, args, callback):
        self.waldorf_client.submit(func, args, callback)

    def map(self, func, iterable):
        self.waldorf_client.map(func, iterable)

    def join(self):
        self.waldorf_client.join()
