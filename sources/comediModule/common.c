/*
 * This is a little helper function to parse options that
 * are common to most of the examples.
 */

#include <stdio.h>
#include <comedilib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <ctype.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include "examples.h"

static char * const default_filename = "/dev/comedi0";

void init_parsed_options(struct parsed_options *options)
{
	memset(options, 0, sizeof(struct parsed_options));
	options->filename = default_filename;
	options->aref = AREF_GROUND;
	options->n_chan = 4;
	options->n_scan = 1000;
	options->freq = 1000.0;
	options->physical = 0;
	options->value = 0.;
}

int parse_options(struct parsed_options *options, int argc, char *argv[])
{
	int c;

	while (-1 != (c = getopt(argc, argv, "a:c:s:r:f:n:N:F:pvdgom"))) {
		switch (c) {
		case 'f':
			options->filename = optarg;
			break;
		case 's':
			options->subdevice = strtoul(optarg, NULL, 0);
			break;
		case 'c':
			options->channel = strtoul(optarg, NULL, 0);
			break;
		case 'a':
			options->aref = strtoul(optarg, NULL, 0);
			break;
		case 'r':
			options->range = strtoul(optarg, NULL, 0);
			break;
		case 'n':
			options->n_chan = strtoul(optarg, NULL, 0);
			break;
		case 'N':
			options->n_scan = strtoul(optarg, NULL, 0);
			break;
		case 'F':
			options->freq = strtod(optarg, NULL);
			break;
		case 'p':
			options->physical = 1;
			break;
		case 'v':
			++options->verbose;
			break;
		case 'd':
			options->aref = AREF_DIFF;
			break;
		case 'g':
			options->aref = AREF_GROUND;
			break;
		case 'o':
			options->aref = AREF_OTHER;
			break;
		case 'm':
			options->aref = AREF_COMMON;
			break;
		default:
			printf("bad option\n");
			exit(1);
		}
	}
	if(optind < argc) {
		/* data value */
		options->value = strtod(argv[optind++], NULL);
	}

	return argc;
}

char *cmd_src(int src,char *buf)
{
	buf[0]=0;

	if(src&TRIG_NONE)strcat(buf,"none|");
	if(src&TRIG_NOW)strcat(buf,"now|");
	if(src&TRIG_FOLLOW)strcat(buf, "follow|");
	if(src&TRIG_TIME)strcat(buf, "time|");
	if(src&TRIG_TIMER)strcat(buf, "timer|");
	if(src&TRIG_COUNT)strcat(buf, "count|");
	if(src&TRIG_EXT)strcat(buf, "ext|");
	if(src&TRIG_INT)strcat(buf, "int|");
#ifdef TRIG_OTHER
	if(src&TRIG_OTHER)strcat(buf, "other|");
#endif

	if(strlen(buf)==0){
		sprintf(buf,"unknown(0x%08x)",src);
	}else{
		buf[strlen(buf)-1]=0;
	}

	return buf;
}

void dump_cmd(FILE *out,comedi_cmd *cmd)
{
	char buf[100];

	fprintf(out,"subdevice:      %d\n",
		cmd->subdev);

	fprintf(out,"start:      %-8s %d\n",
		cmd_src(cmd->start_src,buf),
		cmd->start_arg);

	fprintf(out,"scan_begin: %-8s %d\n",
		cmd_src(cmd->scan_begin_src,buf),
		cmd->scan_begin_arg);

	fprintf(out,"convert:    %-8s %d\n",
		cmd_src(cmd->convert_src,buf),
		cmd->convert_arg);

	fprintf(out,"scan_end:   %-8s %d\n",
		cmd_src(cmd->scan_end_src,buf),
		cmd->scan_end_arg);

	fprintf(out,"stop:       %-8s %d\n",
		cmd_src(cmd->stop_src,buf),
		cmd->stop_arg);
}
