/* Compilation:
   gcc dctdump.c -ljpeg
*/



#include <stdio.h>
#include <assert.h>
#include <setjmp.h>
#include <jpeglib.h>

#define MAX(a,b) (a>=b?a:b);
#define MIN(a,b) (a<=b?a:b);

// {{{ print_dct_coef()
void print_dct_coef(
	j_decompress_ptr cinfo, jvirt_barray_ptr *coeffs, int ci, int opt)
{
	jpeg_component_info *ci_ptr = &cinfo->comp_info[ci];

	JBLOCKARRAY buf =
		(cinfo->mem->access_virt_barray)
		(
		 (j_common_ptr)cinfo,
		 coeffs[ci],
		 0,
		 ci_ptr->v_samp_factor,
		 FALSE
		);

	int sf;
	for (sf = 0; (JDIMENSION)sf < ci_ptr->height_in_blocks; ++sf)
	{
		JDIMENSION b;
		for (b = 0; b < ci_ptr->width_in_blocks; ++b)
		{
			if(opt==1) // format
				printf("\n** DCT block, channel %d **\n", ci);

			int j;
			for(j=0; j<64; j++)
			{
				int dc = 0;
				dc = buf[sf][b][j];
				dc = MAX(-128, dc);
				dc = MIN(128, dc);
				
				if(opt==1)
					printf("%6d", dc);
				else
					printf("%d\n", dc);
			
				if(opt==1)
            	if(j%8==7) 
						puts("");

			}
		}
	}
}
// }}}

// {{{ read_JPEG_file()
GLOBAL(int) read_JPEG_file (char * filename, int opt)
{
	struct jpeg_decompress_struct cinfo;

	struct jpeg_error_mgr jerr;
	FILE * infile;

	if ((infile = fopen(filename, "rb")) == NULL) 
	{
		fprintf(stderr, "can't open %s\n", filename);
		return 0;
	}

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	(void) jpeg_read_header(&cinfo, TRUE);
	jvirt_barray_ptr *coeffs = jpeg_read_coefficients(&cinfo);

	int c=0;
	for(c=0; c<cinfo.num_components; c++)
		print_dct_coef(&cinfo, coeffs, c, opt);

	(void) jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);
	return 1;
}
// }}}

int main(int argc, char **argv)
{
	int ret = 0;
	int opt=0;
	if (argc != 3)
	{
		fprintf(stderr, "usage: %s <jpg file> <raw|format>\n", argv[0]);
		return 1;
	}

	if(strcmp(argv[2], "format")==0)
		opt=1;
	if(strcmp(argv[2], "raw")==0)
		opt=0;

	ret = read_JPEG_file(argv[1], opt);

	return 0;
}


