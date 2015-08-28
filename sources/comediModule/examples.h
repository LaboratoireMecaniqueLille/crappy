
#ifndef _EXAMPLES_H
#define _EXAMPLES_H

#include <stdio.h>

/*
 * Definitions of some of the common code.
 */

extern comedi_t *device;

struct parsed_options
{
	char *filename;
	double value;
	int subdevice;
	int channel;
	int aref;
	int range;
	int physical;
	int verbose;
	int n_chan;
	int n_scan;
	double freq;
};

extern void init_parsed_options(struct parsed_options *options);
extern int parse_options(struct parsed_options *options, int argc, char *argv[]);
extern char *cmd_src(int src,char *buf);
extern void dump_cmd(FILE *file,comedi_cmd *cmd);

#define sec_to_nsec(x) ((x)*1000000000)
#define sec_to_usec(x) ((x)*1000000)
#define sec_to_msec(x) ((x)*1000)
#define msec_to_nsec(x) ((x)*1000000)
#define msec_to_usec(x) ((x)*1000)
#define usec_to_nsec(x) ((x)*1000)

#endif

