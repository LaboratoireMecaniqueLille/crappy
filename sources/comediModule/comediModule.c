#include "comediModule.h"
#include "examples.h"

static PyObject*
comediModule(PyObject* self, PyObject* args)
{
    char *device;
    int subdevice;
    int channel;
    int range;
    int aref;
    int n_chan;
    int n_scan;
    float freq;
    if (!PyArg_ParseTuple(args, "siiiiiif", &device, &subdevice, &channel, &range, &aref, &n_chan, &n_scan, &freq))
        return NULL;
    
    comedi_t *dev;
    comedi_cmd c,*cmd=&c;
    int ret;
    int total=0;
    int i;
    struct timeval start,end;
    int subdev_flags;
    lsampl_t raw;
    struct parsed_options options;
//    printf("filename: %s, subdevice: %i, channel: %i, range: %i, aref: %i, n_chan: %i, n_scan: %i, freq: %f. \n", device, subdevice, channel, range, aref, n_chan, n_scan, freq);
    
    
    
//    printf("filename: %s, value: %d, subdevice: %i, channel: %i, aref: %i, range: %i, physical: %i, verbose: %i, n_chan: %i, n_scan: %i, freq: %d. \n", options.filename, options.value, options.subdevice, options.channel, options.aref, options.range, options.physical, options.verbose, options.n_chan, options.n_scan, options.freq);
    init_parsed_options(&options);
    
    options.filename = device;
    options.subdevice = subdevice;
    options.channel = channel;
    options.range = range;
    options.aref = aref;
    options.n_chan = n_chan;
    options.n_scan = n_scan;
    options.freq = freq;
    
//    printf("filename: %s, value: %d, subdevice: %i, channel: %i, aref: %i, range: %i, physical: %i, verbose: %i, n_chan: %i, n_scan: %i, freq: %d. \n", options.filename, options.value, options.subdevice, options.channel, options.aref, options.range, options.physical, options.verbose, options.n_chan, options.n_scan, options.freq);

    /* open the device */
    dev = comedi_open(options.filename);
    if(!dev){
            comedi_perror(options.filename);
            exit(1);
    }

    // Print numbers for clipped inputs
    comedi_set_global_oor_behavior(COMEDI_OOR_NUMBER);

    /* Set up channel list */
    for(i = 0; i < options.n_chan; i++){
            chanlist[i] = CR_PACK(options.channel + i, options.range, options.aref);
            range_info[i] = comedi_get_range(dev, options.subdevice, options.channel, options.range);
            maxdata[i] = comedi_get_maxdata(dev, options.subdevice, options.channel);
    }
    prepare_cmd_lib(dev, options.subdevice, n_scan, options.n_chan, 1e9 / freq, cmd);

    fprintf(stderr, "command before testing:\n");
    dump_cmd(stderr, cmd);

    ret = comedi_command_test(dev, cmd);
    if(ret < 0){
            comedi_perror("comedi_command_test");
            if(errno == EIO){
                    fprintf(stderr,"Ummm... this subdevice doesn't support commands\n");
            }
            exit(1);
    }
    fprintf(stderr,"first test returned %d (%s)\n", ret,
                    cmdtest_messages[ret]);
    dump_cmd(stderr, cmd);

    ret = comedi_command_test(dev, cmd);
    if(ret < 0){
            comedi_perror("comedi_command_test");
            exit(1);
    }
    fprintf(stderr,"second test returned %d (%s)\n", ret,
                    cmdtest_messages[ret]);
    if(ret!=0){
            dump_cmd(stderr, cmd);
            fprintf(stderr, "Error preparing command\n");
            exit(1);
    }

    /* this is only for informational purposes */
    gettimeofday(&start, NULL);
    fprintf(stderr,"start time: %ld.%06ld\n", start.tv_sec, start.tv_usec);

    /* start the command */
    ret = comedi_command(dev, cmd);
    if(ret < 0){
            comedi_perror("comedi_command");
// 		exit(1);
    }
    subdev_flags = comedi_get_subdevice_flags(dev, options.subdevice);

    const int ndim = 2;
    npy_intp nd[2] = {n_scan, n_chan};
    
    PyObject *myarray;
    double *array_buffer;
    myarray = PyArray_SimpleNew(ndim, nd, NPY_DOUBLE);
    Py_INCREF(myarray);
    array_buffer = (double *)PyArray_DATA(myarray);
    
    while(1){
            ret = read(comedi_fileno(dev),buf,BUFSZ);
            if(ret < 0){
                    /* some error occurred */
                    printf("ERROR: read\n");
                    perror("read");
                    break;
            }else if(ret == 0){
                    /* reached stop condition */
                    break;
            }else{
                    static int col = 0;
                    double j = 0.0;
                    
                    int bytes_per_sample;
                    total += ret;
                    if(options.verbose)fprintf(stderr, "read %d %d\n", ret, total);
                    if(subdev_flags & SDF_LSAMPL)
                            bytes_per_sample = sizeof(lsampl_t);
                    else
                            bytes_per_sample = sizeof(sampl_t);
                    for(i = 0; i < ret / bytes_per_sample; i++){
                            if(subdev_flags & SDF_LSAMPL) {
                                    raw = ((lsampl_t *)buf)[i];
                            } else {
                                    raw = ((sampl_t *)buf)[i];
                            }
                            *array_buffer++ = print_datum(raw, col, 1);
                            col++;
                            if(col == options.n_chan){
                                    col=0;
                                    j++;
                                    printf("\n");
                            }
                    }
            }
    }

    /* this is only for informational purposes */
    gettimeofday(&end,NULL);
    fprintf(stderr,"end time: %ld.%06ld\n", end.tv_sec, end.tv_usec);

    end.tv_sec -= start.tv_sec;
    if(end.tv_usec < start.tv_usec){
            end.tv_sec--;
            end.tv_usec += 1000000;
    }
    end.tv_usec -= start.tv_usec;
    fprintf(stderr,"time: %ld.%06ld\n", end.tv_sec, end.tv_usec);

    return myarray;
}

static PyMethodDef readMethods[] =
{
     {"comedi_read",  (PyCFunction)comediModule, METH_VARARGS, " read many values on comedi device (USB-DUXfast)."},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initcomediModule(void)
{
     (void) Py_InitModule("comediModule", readMethods);
	 import_array();
}


/*
 * This prepares a command in a pretty generic way.  We ask the
 * library to create a stock command that supports periodic
 * sampling of data, then modify the parts we want. */
int prepare_cmd_lib(comedi_t *dev, int subdevice, int n_scan, int n_chan, unsigned scan_period_nanosec, comedi_cmd *cmd)
{
	int ret;

	memset(cmd,0,sizeof(*cmd));

	/* This comedilib function will get us a generic timed
	 * command for a particular board.  If it returns -1,
	 * that's bad. */
	ret = comedi_get_cmd_generic_timed(dev, subdevice, cmd, n_chan, scan_period_nanosec);
	if(ret<0){
		printf("comedi_get_cmd_generic_timed failed\n");
		return ret;
	}

	/* Modify parts of the command */
	cmd->chanlist = chanlist;
	cmd->chanlist_len = n_chan;
	if(cmd->stop_src == TRIG_COUNT) cmd->stop_arg = n_scan;

	return 0;
}

/*
 * Set up a command by hand.  This will not work on some devices.
 * There is no single command that will work on all devices.
 */
int prepare_cmd(comedi_t *dev, int subdevice, int n_scan, int n_chan, unsigned period_nanosec, comedi_cmd *cmd)
{
	memset(cmd,0,sizeof(*cmd));

	/* the subdevice that the command is sent to */
	cmd->subdev =	subdevice;

	/* flags */
	cmd->flags = 0;

	/* Wake up at the end of every scan */
	//cmd->flags |= TRIG_WAKE_EOS;

	/* Use a real-time interrupt, if available */
	//cmd->flags |= TRIG_RT;

	/* each event requires a trigger, which is specified
	   by a source and an argument.  For example, to specify
	   an external digital line 3 as a source, you would use
	   src=TRIG_EXT and arg=3. */

	/* The start of acquisition is controlled by start_src.
	 * TRIG_NOW:     The start_src event occurs start_arg nanoseconds
	 *               after comedi_command() is called.  Currently,
	 *               only start_arg=0 is supported.
	 * TRIG_FOLLOW:  (For an output device.)  The start_src event occurs
	 *               when data is written to the buffer.
	 * TRIG_EXT:     start event occurs when an external trigger
	 *               signal occurs, e.g., a rising edge of a digital
	 *               line.  start_arg chooses the particular digital
	 *               line.
	 * TRIG_INT:     start event occurs on a Comedi internal signal,
	 *               which is typically caused by an INSN_TRIG
	 *               instruction.
	 */
	cmd->start_src =	TRIG_NOW;
	cmd->start_arg =	0;

	/* The timing of the beginning of each scan is controlled by
	 * scan_begin.
	 * TRIG_TIMER:   scan_begin events occur periodically.
	 *               The time between scan_begin events is
	 *               convert_arg nanoseconds.
	 * TRIG_EXT:     scan_begin events occur when an external trigger
	 *               signal occurs, e.g., a rising edge of a digital
	 *               line.  scan_begin_arg chooses the particular digital
	 *               line.
	 * TRIG_FOLLOW:  scan_begin events occur immediately after a scan_end
	 *               event occurs.
	 * The scan_begin_arg that we use here may not be supported exactly
	 * by the device, but it will be adjusted to the nearest supported
	 * value by comedi_command_test(). */
	cmd->scan_begin_src =	TRIG_TIMER;
	cmd->scan_begin_arg = period_nanosec;		/* in ns */

	/* The timing between each sample in a scan is controlled by convert.
	 * TRIG_TIMER:   Conversion events occur periodically.
	 *               The time between convert events is
	 *               convert_arg nanoseconds.
	 * TRIG_EXT:     Conversion events occur when an external trigger
	 *               signal occurs, e.g., a rising edge of a digital
	 *               line.  convert_arg chooses the particular digital
	 *               line.
	 * TRIG_NOW:     All conversion events in a scan occur simultaneously.
	 * Even though it is invalid, we specify 1 ns here.  It will be
	 * adjusted later to a valid value by comedi_command_test() */
	cmd->convert_src =	TRIG_TIMER;
	cmd->convert_arg =	1;		/* in ns */

	/* The end of each scan is almost always specified using
	 * TRIG_COUNT, with the argument being the same as the
	 * number of channels in the chanlist.  You could probably
	 * find a device that allows something else, but it would
	 * be strange. */
	cmd->scan_end_src =	TRIG_COUNT;
	cmd->scan_end_arg =	n_chan;		/* number of channels */

	/* The end of acquisition is controlled by stop_src and
	 * stop_arg.
	 * TRIG_COUNT:  stop acquisition after stop_arg scans.
	 * TRIG_NONE:   continuous acquisition, until stopped using
	 *              comedi_cancel()
	 * */
	cmd->stop_src =		TRIG_COUNT;
	cmd->stop_arg =		n_scan;

	/* the channel list determined which channels are sampled.
	   In general, chanlist_len is the same as scan_end_arg.  Most
	   boards require this.  */
	cmd->chanlist =		chanlist;
	cmd->chanlist_len =	n_chan;

	return 0;
}

double print_datum(lsampl_t raw, int channel_index, short physical) {
	double physical_value;
	if(!physical) {
		printf("%d ",raw);
	} else {
	physical_value = comedi_to_phys(raw, range_info[channel_index], maxdata[channel_index]);
	printf("%f ",physical_value);
	printf("%f(f) ", physical_value);
	return physical_value;
	}
}