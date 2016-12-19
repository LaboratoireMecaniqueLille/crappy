import crappy
channels = ['AIN0', 'AIN1', 'AIN2']
labjack = crappy.technical.LabJack(sensor={'mode': 'streamer', 'scan_rate_per_channel': 10000, 'scans_per_read': 1000, 'channels': channels})
streamer = crappy.blocks.Streamer(sensor=labjack, mean=100)
grapher = crappy.blocks.Grapher([('time(sec)', x) for x in channels], length=10)

crappy.link(streamer, grapher)

crappy.start()
