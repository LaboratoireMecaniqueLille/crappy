import time
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

# conversion : 1 speed = 0.002 mm/s

t0 = time.time()
axes = []
if __name__ == '__main__':
    try:
        # Creating objects

        # sensor = crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3],
        #                                      gain=[1, 1, 1, 1], offset=[0.1723, 0.155, -0.005, 0.005])
        sensor = crappy2.sensor.DummySensor(channels=[0, 1, 2, 3],
                                            gain=[1, 100, 1, 1], offset=[0.1723, 0.155, -0.005, 0.005])

        stream = crappy2.blocks.MeasureByStep(sensor, labels=['t(s)', 'AIN0', 'AIN1', 'AIN2', 'AIN3'], freq=300)

        # motor1 = crappy2.technical.Motion(motor_name="oriental", baudrate=115200, port='/dev/ttyUSB1', num_device=1)
        # motor2 = crappy2.technical.Motion(motor_name="oriental", baudrate=115200, port='/dev/ttyUSB0', num_device=3)
        # motor3 = crappy2.technical.Motion(motor_name="oriental", baudrate=115200, port='/dev/ttyUSB2', num_device=2)
        # motor4 = crappy2.technical.Motion(motor_name="oriental", baudrate=115200, port='/dev/ttyUSB3', num_device=4)

        motor1 = crappy2.technical.Motion(motor_name="DummyTechnical")
        motor2 = crappy2.technical.Motion(motor_name="DummyTechnical")
        motor3 = crappy2.technical.Motion(motor_name="DummyTechnical")
        motor4 = crappy2.technical.Motion(motor_name="DummyTechnical")

        axes = [motor1, motor2, motor3, motor4]
        # axes = [motor1, motor2]
        # Creating blocks

        compacter_effort = crappy2.blocks.Compacter(50)
        save_effort = crappy2.blocks.Saver("/home/biaxe/Bureau/Annie/effort.txt")
        grapher = crappy2.blocks.Grapher(('t(s)', 'AIN0'), ('t(s)', 'AIN1'), ('t(s)', 'AIN2'), ('t(s)', 'AIN3'),
                                         length=2)

        signalGenerator_x = crappy2.blocks.SignalGenerator(
            path=[{"waveform": "limit", "gain": 1, "cycles": 0.5, "phase": 0, "lower_limit": [0.001, 'AIN0'],
                   "upper_limit": [9.9, 'AIN0']}],
            send_freq=400, repeat=True, labels=['t(s)', 'signal'])

        signalGenerator_y = crappy2.blocks.SignalGenerator(
            path=[{"waveform": "limit", "gain": 1, "cycles": 0.5, "phase": 0, "lower_limit": [0.001, 'AIN1'],
                   "upper_limit": [9.9, 'AIN1']}],
            send_freq=400, repeat=True, labels=['t(s)', 'signal'])

        command_axe_x = crappy2.blocks.CommandBiaxe(biaxe_technicals=[motor1, motor2], speed=20)  # vertical
        command_axe_y = crappy2.blocks.CommandBiaxe(biaxe_technicals=[motor3, motor4], speed=20)
        # horizontal # speed must be <0 for traction


        # Creating links

        leffort_to_signal_generator_x = crappy2.links.Link(name="leffort_to_signal_generator_x")
        stream_to_signalGenerator = crappy2.links.Link(name="stream_to_signalGenerator")
        stream_to_compacter = crappy2.links.Link(name="stream_to_compacter")
        compater_to_saver = crappy2.links.Link(name="compater_to_saver")
        compacter_to_grapher = crappy2.links.Link(name="compacter_to_grapher")
        signalGenerator_x_to_command_axe_x = crappy2.links.Link(name="signalGenerator_x_to_command_axe_x")
        signalGenerator_y_to_command_axe_y = crappy2.links.Link(name="signalGenerator_y_to_command_axe_y")

        # Linking objects

        stream.add_output(leffort_to_signal_generator_x)
        stream.add_output(stream_to_signalGenerator)
        stream.add_output(stream_to_compacter)

        signalGenerator_x.add_input(leffort_to_signal_generator_x)
        signalGenerator_x.add_output(signalGenerator_x_to_command_axe_x)

        signalGenerator_y.add_input(stream_to_signalGenerator)
        signalGenerator_y.add_output(signalGenerator_y_to_command_axe_y)

        command_axe_x.add_input(signalGenerator_x_to_command_axe_x)
        command_axe_y.add_input(signalGenerator_y_to_command_axe_y)

        compacter_effort.add_input(stream_to_compacter)
        compacter_effort.add_output(compater_to_saver)
        compacter_effort.add_output(compacter_to_grapher)

        save_effort.add_input(compater_to_saver)

        grapher.add_input(compacter_to_grapher)

        # Starting objects
        # print "top :",crappy2.blocks._meta.MasterBlock.instances
        t0 = time.time()
        for instance in crappy2.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy2.blocks.MasterBlock.instances:
            instance.start()

    # Waiting for execution


    # Stopping objects

    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy2.blocks.MasterBlock.instances:
            try:
                instance.stop()
            except:
                pass
        for axe in axes:
            try:
                axe.close()
            except Exception as e:
                print e
