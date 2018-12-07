#include <stdio.h> 
#include <float.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include <simd.h>
#include <string.h>
#include <limits.h>
#include <assert.h>


#define min(a,b) ((a)>(b)?(b):(a))
#define max(a,b) ((a)>(b)?(a):(b))

typedef float Type;
typedef struct _tagSlavePoolingParam_f
{
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nThreadsNum,nLeftThreadsNum;
	int nBottomOffset,nTopOffset,use_top_mask;
	int  *pMask;
	float *pTopData,*pBottomData,*pTopMask;
}SlavePoolingParam_f;


__thread_local_fix  dma_desc pool_dmaget2,dmaputmask,pool_dmaput2;


void poolingForwardSmallMax_f(SlavePoolingParam_f *pParam)
{
	const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,nRows,nPoolIndex,nBottomIndex;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	

	int sumOfImage,myBatchSize,myCount,myLeft,myBatchLeft,myIter;
	Type *mypTopData,*mypBottomData,*mypTopMask;
	int  *mypMask;
	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;


	sumOfImage = nCount *64 + nLeftMaxThreadsNum;
	myBatchSize = ((28*28-1)/(height_*width_))+1;
	myCount = (sumOfImage / myBatchSize)/64;
	myBatchLeft = (sumOfImage / myBatchSize) - myCount * 64;
	myLeft = sumOfImage % myBatchSize;

	// if(myid==0) printf("sumofimage=%d\n",sumOfImage);
	// if(myid==0) printf("mybatchsize=%d\n",myBatchSize);
	// if(myid==0) printf("mycount=%d\n",myCount);
	// if(myid==0) printf("mybatchleft=%d\n",myBatchLeft);
	// if(myid==0) printf("myleft=%d\n",myLeft);
	// if(myid==0) printf("maxthread=%d\n",nMaxThreadsNum);
	
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,dmaputmask,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	dma_set_op(&dmaputmask, DMA_PUT);
	dma_set_mode(&dmaputmask, PE_MODE);
	dma_set_reply(&dmaputmask, &putmaskreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;


	// if(myid==0) printf("ntopsize=%d\n",nTopSize);
	// if(myid==0) printf("nbottomsize=%d\n",nBottomSize);


	if(use_top_mask>0)
	  nMaskSize = nTopSize;
    else
	  nMaskSize = pooled_height_ * pooled_width_*sizeof(int);  	
	//init finish

	pTopData  = (Type*)(long)ldm_malloc(nTopSize * myBatchSize);
	pBottomData = (Type*)(long)ldm_malloc(nBottomSize * myBatchSize);
			
	if(use_top_mask>0)
	  pTopMask  = (Type*)(long)ldm_malloc(nMaskSize * myBatchSize);
	else
	  pMask  = (int*)(long)ldm_malloc(nMaskSize * myBatchSize); 
		
	dma_set_size(&pool_dmaput2, nTopSize * myBatchSize);
	dma_set_size(&pool_dmaget2, nBottomSize * myBatchSize);
	dma_set_size(&dmaputmask, nMaskSize * myBatchSize);

	// if(myid==0) printf("ptopdata[0]=%f\n",pTopData[0]);


	for(i=0;i<myCount;i++)
	{
		nOffset = (i*nMaxThreadsNum + myid)*myBatchSize;		
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		dma_wait(&getreply,1);getreply=0;
        //for multiple images
        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myBatchSize;++myIter)
        {

		for (ph = 0; ph < pooled_height_; ++ph) {
		  hstart = ph * stride_h_ - pad_h_;
		  hend = min(hstart + kernel_h_, height_);
		  hstart = max(hstart, 0);
	      nPoolIndex = ph*pooled_width_;				
		  for (pw = 0; pw < pooled_width_; ++pw) {
			wstart = pw * stride_w_ - pad_w_;
			wend = min(wstart + kernel_w_, width_);
			wstart = max(wstart, 0);
			pool_index = nPoolIndex + pw;
			mypTopData[pool_index] = -FLT_MAX;
					
			for (h = hstart; h < hend; ++h) {
			  nBottomIndex = h * width_;
			  for (w = wstart; w < wend; ++w) {
				index = nBottomIndex + w;
				if (mypBottomData[index] > mypTopData[pool_index]) {
				  mypTopData[pool_index] = mypBottomData[index];						  
				  if(use_top_mask>0) 
	  				mypTopMask[pool_index] = index;  
				  else
		  			mypMask[pool_index] = index;
				}
			  }
			}
		  }
		}
		mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;

	    }
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));
		if(use_top_mask>0) 
			dma(dmaputmask,(long)(pParam->pTopMask+nOffset*nTopOffset),(long)(pTopMask));
		else
			dma(dmaputmask,(long)(pParam->pMask+nOffset*nTopOffset),(long)(pMask));
		
		dma_wait(&putreply,1);putreply=0;				
		dma_wait(&putmaskreply,1);putmaskreply=0;	
	}

	// if(myid==0) printf("ptopdata[0]=%f\n",pTopData[0]);

	if(myBatchLeft >0 && myid < myBatchLeft)
	{
		nOffset = (myCount*nMaxThreadsNum + myid)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		dma_wait(&getreply,1);getreply=0;
		mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;
        for(myIter=0;myIter<myBatchSize;++myIter)
        {
		        for (ph = 0; ph < pooled_height_; ++ph) {
		          hstart = ph * stride_h_ - pad_h_;
		          hend = min(hstart + kernel_h_, height_);
		          hstart = max(hstart, 0);
	              nPoolIndex = ph*pooled_width_;				
		          for (pw = 0; pw < pooled_width_; ++pw) {
		        	wstart = pw * stride_w_ - pad_w_;
		        	wend = min(wstart + kernel_w_, width_);
		        	wstart = max(wstart, 0);
		        	pool_index = nPoolIndex + pw;
		        	mypTopData[pool_index] = -FLT_MAX;					
		        	for (h = hstart; h < hend; ++h) {
		        	  nBottomIndex = h * width_;
		        	  for (w = wstart; w < wend; ++w) {
		        		index = nBottomIndex + w;
		        		if (mypBottomData[index] > mypTopData[pool_index]) {
		        		  mypTopData[pool_index] = mypBottomData[index];						  
		        		  if(use_top_mask>0) 
		        			mypTopMask[pool_index] = index;  
		        		  else
		        			mypMask[pool_index] = index;
		        		}
		        	  }
		        	}
		          }
		        }
		        mypBottomData+=height_*width_;
		        mypTopData+=pooled_height_*pooled_width_;
		        mypTopMask++;
		        mypMask+=pooled_height_*pooled_width_;
						// if(myid==0) printf("ptopdata[0]=%f\n",pTopData[0]);
	    }
			// if(myid==0) printf("ptopdata[0]=%f\n",pTopData[0]);
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));
		if(use_top_mask>0) 
			dma(dmaputmask,(long)(pParam->pTopMask+nOffset*nTopOffset),(long)(pTopMask));
		else
			dma(dmaputmask,(long)(pParam->pMask+nOffset*nTopOffset),(long)(pMask));
			
		dma_wait(&putreply,1);putreply=0;				
		dma_wait(&putmaskreply,1);putmaskreply=0;		
	}	


	// sumOfImage = nCount *64 + nLeftMaxThreadsNum;
	// myBatchSize = ((28*28-1)/(height_*width_))+1;
	// myCount = (sumOfImage / myBatchSize)/64;
	// myBatchLeft = (sumOfImage / myBatchSize) - myCount * 64;
	// myLeft = sumOfImage % myBatchSize;

	if(myLeft > 0 && myid == 0)
	{
		dma_set_size(&pool_dmaput2, nTopSize * myLeft);
		dma_set_size(&pool_dmaget2, nBottomSize * myLeft);
		dma_set_size(&dmaputmask, nMaskSize * myLeft);

		nOffset = (myCount*nMaxThreadsNum + myBatchLeft)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		dma_wait(&getreply,1);getreply=0;
		mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;
        for(myIter=0;myIter<myLeft;++myIter)
        {
		for (ph = 0; ph < pooled_height_; ++ph) {
		  hstart = ph * stride_h_ - pad_h_;
		  hend = min(hstart + kernel_h_, height_);
		  hstart = max(hstart, 0);
	      nPoolIndex = ph*pooled_width_;				
		  for (pw = 0; pw < pooled_width_; ++pw) {
			wstart = pw * stride_w_ - pad_w_;
			wend = min(wstart + kernel_w_, width_);
			wstart = max(wstart, 0);
			pool_index = nPoolIndex + pw;
			mypTopData[pool_index] = -FLT_MAX;					
			for (h = hstart; h < hend; ++h) {
			  nBottomIndex = h * width_;
			  for (w = wstart; w < wend; ++w) {
				index = nBottomIndex + w;
				if (mypBottomData[index] > mypTopData[pool_index]) {
				  mypTopData[pool_index] = mypBottomData[index];						  
				  if(use_top_mask>0) 
					mypTopMask[pool_index] = index;  
				  else
					mypMask[pool_index] = index;
				}
			  }
			}
		  }
		}
		mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
	    }
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));
		if(use_top_mask>0) 
			dma(dmaputmask,(long)(pParam->pTopMask+nOffset*nTopOffset),(long)(pTopMask));
		else
			dma(dmaputmask,(long)(pParam->pMask+nOffset*nTopOffset),(long)(pMask));
			
		dma_wait(&putreply,1);putreply=0;				
		dma_wait(&putmaskreply,1);putmaskreply=0;		



	}


	ldm_free(pTopData,nTopSize * myBatchSize);
	ldm_free(pBottomData,nBottomSize * myBatchSize);
	if(use_top_mask>0)
		ldm_free(pTopMask,nMaskSize * myBatchSize);
	else
		ldm_free(pMask,nMaskSize * myBatchSize);
}


void poolingForwardSmallAvg_f(SlavePoolingParam_f *pParam)
{
	const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,nRows,nPoolIndex,pool_size;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask,dSum=0;	
	int  *pMask;	

    int sumOfImage,myBatchSize,myCount,myLeft,myBatchLeft,myIter;
	Type *mypTopData,*mypBottomData,*mypTopMask;
	int  *mypMask;

	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;

    sumOfImage = nCount *64 + nLeftMaxThreadsNum;
	myBatchSize = ((28*28-1)/(height_*width_))+1;
	myCount = (sumOfImage / myBatchSize)/64;
	myBatchLeft = (sumOfImage / myBatchSize) - myCount * 64;
	myLeft = sumOfImage % myBatchSize;
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;

    pTopData  = (Type*)(long)ldm_malloc(nTopSize * myBatchSize);
	pBottomData = (Type*)(long)ldm_malloc(nBottomSize * myBatchSize);
		
	dma_set_size(&pool_dmaget2, nBottomSize * myBatchSize);
	dma_set_size(&pool_dmaput2, nTopSize * myBatchSize);
		
	for(i=0;i<myCount;i++)
	{
		nOffset = (i*nMaxThreadsNum + myid) * myBatchSize;		
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		memset(pTopData,0,nTopSize * myBatchSize);
		dma_wait(&getreply,1);getreply=0;	

        mypBottomData=pBottomData;
        mypTopData=pTopData;			
			
        for(myIter=0; myIter<myBatchSize;++myIter)
        {
		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph*pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				dSum = 0;
				for (h = hstart; h < hend; ++h) {
				  for ( w = wstart; w < wend; ++w) {
					dSum +=	mypBottomData[h * width_ + w];
				  }
			    }
			    mypTopData[nPoolIndex + pw] = dSum/pool_size;				
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));			
		dma_wait(&putreply,1);putreply=0;				
	}
		//Left data process		
	if(myBatchLeft >0 && myid < myBatchLeft)
	{
		nOffset = (myCount*nMaxThreadsNum + myid)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		memset(pTopData,0,nTopSize*myBatchSize);
		dma_wait(&getreply,1);getreply=0;
        mypBottomData=pBottomData;
        mypTopData=pTopData;
        for(myIter=0;myIter<myBatchSize;++myIter)
        {				
			
		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph*pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				dSum = 0;
				for ( h = hstart; h < hend; ++h) {
				  for ( w = wstart; w < wend; ++w) {
					dSum +=	mypBottomData[h * width_ + w];
				  }
			    }
			    mypTopData[nPoolIndex + pw] = dSum/pool_size;				
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));			
		dma_wait(&putreply,1);putreply=0;			
	}

    if(myLeft >0 && myid==0)
    {
        dma_set_size(&pool_dmaget2, nBottomSize * myLeft);
	    dma_set_size(&pool_dmaput2, nTopSize * myLeft);

        nOffset = (myCount*nMaxThreadsNum + myBatchLeft)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
		memset(pTopData,0,nTopSize*myLeft);
		dma_wait(&getreply,1);getreply=0;
        mypBottomData=pBottomData;
        mypTopData=pTopData;
        for(myIter=0;myIter<myLeft;++myIter)
        {				
			
		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph*pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				dSum = 0;
				for ( h = hstart; h < hend; ++h) {
				  for ( w = wstart; w < wend; ++w) {
					dSum +=	mypBottomData[h * width_ + w];
				  }
			    }
			    mypTopData[nPoolIndex + pw] = dSum/pool_size;				
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));			
		dma_wait(&putreply,1);putreply=0;	
    }
	ldm_free(pTopData,nTopSize*myBatchSize);
	ldm_free(pBottomData,nBottomSize*myBatchSize);    
}

void poolingBackwardSmallMax_f(SlavePoolingParam_f *pParam)
{
	const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	
  //dma_desc pool_dmaget2,pool_dmaput2;	

    int sumOfImage,myBatchSize,myCount,myLeft,myBatchLeft,myIter;
	Type *mypTopData,*mypBottomData,*mypTopMask;
	int  *mypMask;


	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;

    sumOfImage = nCount *64 + nLeftMaxThreadsNum;
	myBatchSize = ((28*28-1)/(height_*width_))+1;
	myCount = (sumOfImage / myBatchSize)/64;
	myBatchLeft = (sumOfImage / myBatchSize) - myCount * 64;
	myLeft = sumOfImage % myBatchSize;

	
	if(myid >= nMaxThreadsNum) return;	
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
	if(use_top_mask>0)
	  nMaskSize = nTopSize;
    else
	  nMaskSize = pooled_height_ * pooled_width_*sizeof(int);  	
	int nMaxSize = height_*width_-1;

    pTopData  = (Type*)(long)ldm_malloc(nTopSize*myBatchSize);
	pBottomData = (Type*)(long)ldm_malloc(nBottomSize*myBatchSize);
			
	if(use_top_mask>0)
	  pTopMask  = (Type*)(long)ldm_malloc(nMaskSize*myBatchSize);
	else
	  pMask  = (int*)(long)ldm_malloc(nMaskSize*myBatchSize); 
	dma_set_size(&pool_dmaput2, nBottomSize*myBatchSize);
    for(i=0;i<myCount;i++)
	{
		nOffset = (i*nMaxThreadsNum + myid)*myBatchSize;		
		nOffset0 = nOffset * nTopOffset;
		nOffset1 = nOffset * nBottomOffset;
			
		dma_set_size(&pool_dmaget2, nTopSize*myBatchSize);
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0),(long)(pTopData));
		memset(pBottomData,0,nBottomSize*myBatchSize);
		dma_wait(&getreply,1);getreply=0;				
		if(use_top_mask>0) 
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myBatchSize);
			dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0),(long)(pTopMask));
			dma_wait(&getreply,1);getreply=0;	
		}	
		else
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myBatchSize);
		    dma(pool_dmaget2,(long)(pParam->pMask+nOffset0),(long)(pMask));
			dma_wait(&getreply,1);getreply=0;	
		}

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myBatchSize;++myIter)
        {

		for (ph = 0; ph < pooled_height_; ++ph) {
		  pool_index = ph * pooled_width_;
		  for (pw = 0; pw < pooled_width_; ++pw) {
			index = pool_index + pw;
	  		bottom_index =	use_top_mask >0? mypTopMask[index] : mypMask[index];
            if(bottom_index<0 || bottom_index > nMaxSize)continue;
		  	mypBottomData[bottom_index] += mypTopData[index];
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset1),(long)(pBottomData));
		dma_wait(&putreply,1);putreply=0;				
	}

    if(myBatchLeft >0 && myid < myBatchLeft)
	{
		nOffset = (myCount*nMaxThreadsNum + myid)*myBatchSize;
		nOffset0 = nOffset * nTopOffset;
		nOffset1 = nOffset * nBottomOffset;
		dma_set_size(&pool_dmaget2, nTopSize*myBatchSize);
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0),(long)(pTopData));
		memset(pBottomData,0,nBottomSize*myBatchSize);
		dma_wait(&getreply,1);getreply=0;				
		if(use_top_mask>0) 
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myBatchSize);
			dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0),(long)(pTopMask));
			dma_wait(&getreply,1);getreply=0;	
		}	
		else
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myBatchSize);
			dma(pool_dmaget2,(long)(pParam->pMask+nOffset0),(long)(pMask));
			dma_wait(&getreply,1);getreply=0;	
		}

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myBatchSize;++myIter)
        {
		for (ph = 0; ph < pooled_height_; ++ph) {
		  pool_index = ph * pooled_width_;
		  for (pw = 0; pw < pooled_width_; ++pw) {
			  index = pool_index + pw;
			  bottom_index =	use_top_mask >0? mypTopMask[index] : mypMask[index];
			  if(bottom_index<0 || bottom_index > nMaxSize)continue;
			  mypBottomData[bottom_index] += mypTopData[index];
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset1),(long)(pBottomData));
		dma_wait(&putreply,1);putreply=0;		
	}	
    if(myLeft >0 && myid==0)
    {
        dma_set_size(&pool_dmaput2, nBottomSize*myLeft);
        nOffset = (myCount*nMaxThreadsNum + myBatchLeft)*myBatchSize;
		nOffset0 = nOffset * nTopOffset;
		nOffset1 = nOffset * nBottomOffset;
		dma_set_size(&pool_dmaget2, nTopSize*myLeft);
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0),(long)(pTopData));
		memset(pBottomData,0,nBottomSize*myLeft);
		dma_wait(&getreply,1);getreply=0;				
		if(use_top_mask>0) 
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myLeft);
			dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0),(long)(pTopMask));
			dma_wait(&getreply,1);getreply=0;	
		}	
		else
		{
			dma_set_size(&pool_dmaget2, nMaskSize*myLeft);
			dma(pool_dmaget2,(long)(pParam->pMask+nOffset0),(long)(pMask));
			dma_wait(&getreply,1);getreply=0;	
		}

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myLeft;++myIter)
        {
		for (ph = 0; ph < pooled_height_; ++ph) {
		  pool_index = ph * pooled_width_;
		  for (pw = 0; pw < pooled_width_; ++pw) {
			  index = pool_index + pw;
			  bottom_index =	use_top_mask >0? mypTopMask[index] : mypMask[index];
			  if(bottom_index<0 || bottom_index > nMaxSize)continue;
			  mypBottomData[bottom_index] += mypTopData[index];
		  }
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset1),(long)(pBottomData));
		dma_wait(&putreply,1);putreply=0;	

    }
    ldm_free(pTopData,nTopSize*myBatchSize);
	ldm_free(pBottomData,nBottomSize*myBatchSize);
	if(use_top_mask>0)
		ldm_free(pTopMask,nMaskSize*myBatchSize);
	else
		ldm_free(pMask,nMaskSize*myBatchSize);

}

void poolingBackwardSmallAvg_f(SlavePoolingParam_f *pParam)
{
	const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,pool_size,nPoolIndex=0;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	

    int sumOfImage,myBatchSize,myCount,myLeft,myBatchLeft,myIter;
	Type *mypTopData,*mypBottomData,*mypTopMask;
	int  *mypMask;
	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;

    sumOfImage = nCount *64 + nLeftMaxThreadsNum;
	myBatchSize = ((28*28-1)/(height_*width_))+1;
	myCount = (sumOfImage / myBatchSize)/64;
	myBatchLeft = (sumOfImage / myBatchSize) - myCount * 64;
	myLeft = sumOfImage % myBatchSize;
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;

    pTopData  = (Type*)(long)ldm_malloc(nTopSize * myBatchSize);
	pBottomData = (Type*)(long)ldm_malloc(nBottomSize * myBatchSize);
		
	dma_set_size(&pool_dmaget2, nTopSize * myBatchSize);
	dma_set_size(&pool_dmaput2, nBottomSize * myBatchSize);
			
	for(i=0;i<myCount;i++)
	{
		nOffset = (i*nMaxThreadsNum + myid) * myBatchSize;		
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));	
		memset(pBottomData,0,nBottomSize * myBatchSize);
		dma_wait(&getreply,1);getreply=0;	

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myBatchSize;++myIter)
        {			
		
		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph * pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				pool_index = nPoolIndex + pw;
				for ( h = hstart; h < hend; ++h) {
				  bottom_index = h * width_;
				  for ( w = wstart; w < wend; ++w) {
					mypBottomData[bottom_index + w] += mypTopData[pool_index] / pool_size;
				  }
			    }
			}
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
			
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));			
		dma_wait(&putreply,1);putreply=0;				
	}

    if(myBatchLeft >0 && myid < myBatchLeft)
	{
		nOffset = (myCount*nMaxThreadsNum + myid)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));	
		memset(pBottomData,0,nBottomSize*myBatchSize);
		dma_wait(&getreply,1);getreply=0;	

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myBatchSize;++myIter)
        {

		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph * pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				pool_index = nPoolIndex + pw;
				for ( h = hstart; h < hend; ++h) {
				  bottom_index = h * width_;
				  for ( w = wstart; w < wend; ++w) {
					mypBottomData[bottom_index + w] += mypTopData[pool_index] / pool_size;
				  }
			    }
			}
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
		
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));			
		dma_wait(&putreply,1);putreply=0;			
	}

    if(myLeft >0 && myid==0)
    {
        dma_set_size(&pool_dmaget2, nTopSize * myLeft);
	    dma_set_size(&pool_dmaput2, nBottomSize * myLeft);

        nOffset = (myCount*nMaxThreadsNum + myBatchLeft)*myBatchSize;
		dma(pool_dmaget2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));	
		memset(pBottomData,0,nBottomSize*myLeft);
		dma_wait(&getreply,1);getreply=0;	

        mypBottomData=pBottomData;
        mypTopData=pTopData;
        mypTopMask=pTopMask;
        mypMask=pMask;

        for(myIter=0;myIter<myLeft;++myIter)
        {

		for (ph = 0; ph < pooled_height_; ++ph) {
		    hstart = ph * stride_h_ - pad_h_;
			hend = min(hstart + kernel_h_, height_ + pad_h_);
			hstart = max(hstart, 0);
			hend = min(hend, height_);
			nPoolIndex = ph * pooled_width_;
			for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_ + pad_w_);
				wstart = max(wstart, 0);
				wend = min(wend, width_);
				pool_size = (hend - hstart) * (wend - wstart);
				pool_index = nPoolIndex + pw;
				for ( h = hstart; h < hend; ++h) {
				  bottom_index = h * width_;
				  for ( w = wstart; w < wend; ++w) {
					mypBottomData[bottom_index + w] += mypTopData[pool_index] / pool_size;
				  }
			    }
			}
		}
        mypBottomData+=height_*width_;
		mypTopData+=pooled_height_*pooled_width_;
		mypTopMask++;
		mypMask+=pooled_height_*pooled_width_;
        }
		
		dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));			
		dma_wait(&putreply,1);putreply=0;		
    }
}


void poolingBackwardMax_f(SlavePoolingParam_f *pParam)
{
  const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	
  //dma_desc pool_dmaget2,pool_dmaput2;	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;
	
	if(myid >= nMaxThreadsNum) return;	
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
	if(use_top_mask>0)
	  nMaskSize = nTopSize;
  else
	  nMaskSize = pooled_height_ * pooled_width_*sizeof(int);  	
	int nMaxSize = height_*width_-1;

	if((nTopSize+nBottomSize+nMaskSize) > nMaxBuffSize)
	{
	    nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
		int nSplitCount=0,nSplitRows =0,nLeftRows = 0;
		int nTopSize1 = 0,nMaskSize1=0,nBottomIndex;
		nSplitRows = pooled_height_;
        int nKernelSize = kernel_h_ *width_*sizeof(Type);
        int nStartAddr = 0;
		while(nBottomSize <nKernelSize)
		{
			nSplitRows = nSplitRows>>1;			
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			if(use_top_mask >0) nMaskSize = nTopSize;				
			else nMaskSize = nSplitRows*pooled_width_*sizeof(int);
			nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
			nSplitCount++;
		}
		
		if(nSplitCount <1){
			nSplitRows = 0;
			nLeftRows = pooled_height_;
			nTopSize1 = pooled_height_ * pooled_width_*sizeof(Type);
			nMaskSize1 = pooled_height_ * pooled_width_*sizeof(int);
		}
		else{
			nSplitCount = pooled_height_/nSplitRows;
			nLeftRows = pooled_height_%nSplitRows;
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			if(use_top_mask >0) 
			{
				nMaskSize = nTopSize;	
				nMaskSize1 = nLeftRows*pooled_width_*sizeof(Type);
			}
			else
			{
				nMaskSize = nSplitRows*pooled_width_*sizeof(int);
				nMaskSize1 = nLeftRows*pooled_width_*sizeof(int);
			}				
			nTopSize1 = nLeftRows*pooled_width_*sizeof(Type);							
		}
		
		nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
		//if(myid<1)printf("nSplitCount=%d nSplitRows=%d nTopSize=%d nBottomSize=%d\n",nSplitCount,nSplitRows,nTopSize,nBottomSize);
		if(use_top_mask>0)
		{
			nMaskSize = nTopSize;
			pTopMask  = (Type*)(long)ldm_malloc(nMaskSize);
		}
		else
		{
			pMask  = (int*)(long)ldm_malloc(nMaskSize);	
        }	
		
		pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);		
		for(i=0;i<nCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
						
			for(j=0;j<nSplitCount;j++)
			{
				if(use_top_mask>0) 
				{
					dma_set_size(&pool_dmaget2, nTopSize);  				
					dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopMask));
					dma_wait(&getreply,1);getreply=0;
				}	
				else
				{
					dma_set_size(&pool_dmaget2, nMaskSize);  				
					dma(pool_dmaget2,(long)(pParam->pMask+nOffset0+j*nSplitRows*pooled_width_),(long)(pMask));
					dma_wait(&getreply,1);getreply=0;
				}
				
				dma_set_size(&pool_dmaget2, nTopSize);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					pool_index = ph*pooled_width_;
					hstart = INT_MAX,hend = 0;
					if(use_top_mask>0)
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pTopMask[index] >=0 && pTopMask[index]<hstart)
								hstart = pTopMask[index];
							if(pTopMask[index]>hend)
								hend = pTopMask[index];
						}		
					}
					else	
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pMask[index] >=0 && pMask[index]<hstart)
								hstart = pMask[index];
							if(pMask[index]>hend)
								hend = pMask[index];
						}
					}					
          
		  			hend = hend %width_ >0 ? (hend /width_)+1 :hend /width_ ;
					hstart = hstart /width_;
					nKernelSize = (hend - hstart)*width_*sizeof(Type);
					nStartAddr = hstart*width_;
					nBottomIndex = nOffset1+nStartAddr;
					dma_set_size(&pool_dmaget2, nKernelSize);  				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&getreply,1);getreply=0;
					for (pw = 0; pw < pooled_width_; ++pw) {
						
						index = pool_index+pw;
						bottom_index =	(use_top_mask>0 ? pTopMask[index] : pMask[index]) - nStartAddr;
						if(bottom_index<0 || bottom_index >nMaxSize)continue;
						pBottomData[bottom_index] += pTopData[index];
					}
					dma_set_size(&pool_dmaput2, nKernelSize);				
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&putreply,1);putreply=0;
				}
			}
			if(nLeftRows > 0)
			{
				nOffset = nSplitCount * nSplitRows*pooled_width_;
			 	if(use_top_mask>0) 
				{
					dma_set_size(&pool_dmaget2, nTopSize1);  				
					dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
					dma_wait(&getreply,1);getreply=0;
				}	
				else
				{
					dma_set_size(&pool_dmaget2, nMaskSize1);  				
					dma(pool_dmaget2,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
					dma_wait(&getreply,1);getreply=0;
				}
				
				dma_set_size(&pool_dmaget2, nTopSize1);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					pool_index = ph*pooled_width_;
					hstart = INT_MAX,hend = 0;
					if(use_top_mask>0)
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pTopMask[index] >=0 && pTopMask[index]<hstart)
								hstart = pTopMask[index];
							if(pTopMask[index]>hend)
								hend = pTopMask[index];
						}		
					}
					else	
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pMask[index] >=0 && pMask[index]<hstart)
								hstart = pMask[index];
							if(pMask[index]>hend)
								hend = pMask[index];
						}
					}					
				    hend = hend %width_ >0 ? (hend /width_)+1 :hend /width_ ;
					hstart = hstart /width_;					
        			nKernelSize = (hend - hstart)*width_*sizeof(Type);
					
					nStartAddr = hstart*width_;
					nBottomIndex = nOffset1+nStartAddr;
					
					dma_set_size(&pool_dmaget2, nKernelSize);  				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&getreply,1);getreply=0;
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						
						index = pool_index+pw;
						bottom_index =	(use_top_mask>0 ? pTopMask[index] : pMask[index]) - nStartAddr;
						if(bottom_index <0 || bottom_index >nMaxSize) continue;
						pBottomData[bottom_index] += pTopData[index];
					}
					dma_set_size(&pool_dmaput2, nKernelSize);				
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&putreply,1);putreply=0;
				}
			}			
		}	  
				
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;			
			for(j=0;j<nSplitCount;j++)
			{
				if(use_top_mask>0) 
				{
					dma_set_size(&pool_dmaget2, nTopSize);  				
					dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopMask));
					dma_wait(&getreply,1);getreply=0;
				}	
				else
				{
					dma_set_size(&pool_dmaget2, nMaskSize);  				
					dma(pool_dmaget2,(long)(pParam->pMask+nOffset0+j*nSplitRows*pooled_width_),(long)(pMask));
					dma_wait(&getreply,1);getreply=0;
				}
				
				dma_set_size(&pool_dmaget2, nTopSize);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					pool_index = ph*pooled_width_;
					hstart = INT_MAX,hend = 0;
					if(use_top_mask>0)
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pTopMask[index]>=0 && pTopMask[index]<hstart)
								hstart = pTopMask[index];
							if(pTopMask[index]>hend)
								hend = pTopMask[index];
						}		
					}
					else	
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pMask[index]>=0 && pMask[index]<hstart)
								hstart = pMask[index];
							if(pMask[index]>hend)
								hend = pMask[index];
						}
					}					
          
					hend = hend %width_ >0 ? (hend /width_)+1 :hend /width_ ;
					hstart = hstart /width_;
          
					nKernelSize = (hend - hstart)*width_*sizeof(Type);
					
					nStartAddr = hstart*width_;
					nBottomIndex = nOffset1+nStartAddr;
					
					dma_set_size(&pool_dmaget2, nKernelSize);  				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&getreply,1);getreply=0;
					for (pw = 0; pw < pooled_width_; ++pw) {
						
						index = pool_index+pw;
						bottom_index =	(use_top_mask>0 ? pTopMask[index] : pMask[index]) - nStartAddr;
						if(bottom_index<0 || bottom_index > nMaxSize)continue;
						pBottomData[bottom_index] += pTopData[index];
					}
					dma_set_size(&pool_dmaput2, nKernelSize);				
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&putreply,1);putreply=0;
				}
			}
			if(nLeftRows > 0)
			{
				nOffset = nSplitCount * nSplitRows*pooled_width_;
			 	if(use_top_mask>0) 
				{
					dma_set_size(&pool_dmaget2, nTopSize1);  				
					dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
					dma_wait(&getreply,1);getreply=0;
				}	
				else
				{
					dma_set_size(&pool_dmaget2, nMaskSize1);  				
					dma(pool_dmaget2,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
					dma_wait(&getreply,1);getreply=0;
				}
				
				dma_set_size(&pool_dmaget2, nTopSize1);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					pool_index = ph*pooled_width_;
					hstart = INT_MAX,hend = 0;
					if(use_top_mask>0)
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pTopMask[index]>=0 && pTopMask[index]<hstart)
								hstart = pTopMask[index];
							if(pTopMask[index]>hend)
								hend = pTopMask[index];
						}		
					}
					else	
					{
						for(pw=0;pw<pooled_width_;pw++)
						{
							index = pool_index+pw;
							if(pMask[index] >=0 && pMask[index]<hstart)
								hstart = pMask[index];
							if(pMask[index]>hend)
								hend = pMask[index];
						}
					}					
          
					hend = hend %width_ >0 ? (hend /width_)+1 :hend /width_ ;
					hstart = hstart /width_;				
         
					nKernelSize = (hend - hstart)*width_*sizeof(Type);
					
					nStartAddr = hstart*width_;
					nBottomIndex = nOffset1+nStartAddr;
					
					dma_set_size(&pool_dmaget2, nKernelSize);  				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&getreply,1);getreply=0;
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						
						index = pool_index+pw;
						bottom_index =	(use_top_mask>0 ? pTopMask[index] : pMask[index])- nStartAddr;
						if(bottom_index<0 || bottom_index > nMaxSize) continue;
						pBottomData[bottom_index] += pTopData[index];
					}
					dma_set_size(&pool_dmaput2, nKernelSize);				
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));
					dma_wait(&putreply,1);putreply=0;
				}
			}
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
		if(use_top_mask>0)
			ldm_free(pTopMask,nMaskSize);
		else
			ldm_free(pMask,nMaskSize);
	}
	else
	{ 
		pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);
			
		if(use_top_mask>0)
		  pTopMask  = (Type*)(long)ldm_malloc(nMaskSize);
		else
		  pMask  = (int*)(long)ldm_malloc(nMaskSize); 
		dma_set_size(&pool_dmaput2, nBottomSize);
		for(i=0;i<nCount;i++)
		{
			nOffset = i*nMaxThreadsNum + myid;		
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
			
			dma_set_size(&pool_dmaget2, nTopSize);
			dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0),(long)(pTopData));
			memset(pBottomData,0,nBottomSize);
			dma_wait(&getreply,1);getreply=0;				
			if(use_top_mask>0) 
			{
				dma_set_size(&pool_dmaget2, nMaskSize);
				dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0),(long)(pTopMask));
				dma_wait(&getreply,1);getreply=0;	
			}	
			else
			{
				dma_set_size(&pool_dmaget2, nMaskSize);
			dma(pool_dmaget2,(long)(pParam->pMask+nOffset0),(long)(pMask));
				dma_wait(&getreply,1);getreply=0;	
			}
			for (ph = 0; ph < pooled_height_; ++ph) {
			  pool_index = ph * pooled_width_;
			  for (pw = 0; pw < pooled_width_; ++pw) {
	  			index = pool_index + pw;
		  		bottom_index =	use_top_mask >0? pTopMask[index] : pMask[index];
          if(bottom_index<0 || bottom_index > nMaxSize)continue;
			  	pBottomData[bottom_index] += pTopData[index];
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset1),(long)(pBottomData));
			dma_wait(&putreply,1);putreply=0;				
		}
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
			dma_set_size(&pool_dmaget2, nTopSize);
			dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0),(long)(pTopData));
			memset(pBottomData,0,nBottomSize);
			dma_wait(&getreply,1);getreply=0;				
			if(use_top_mask>0) 
			{
				dma_set_size(&pool_dmaget2, nMaskSize);
				dma(pool_dmaget2,(long)(pParam->pTopMask+nOffset0),(long)(pTopMask));
				dma_wait(&getreply,1);getreply=0;	
			}	
			else
			{
				dma_set_size(&pool_dmaget2, nMaskSize);
				dma(pool_dmaget2,(long)(pParam->pMask+nOffset0),(long)(pMask));
				dma_wait(&getreply,1);getreply=0;	
			}
			for (ph = 0; ph < pooled_height_; ++ph) {
			  pool_index = ph * pooled_width_;
			  for (pw = 0; pw < pooled_width_; ++pw) {
				  index = pool_index + pw;
				  bottom_index =	use_top_mask >0? pTopMask[index] : pMask[index];
				  if(bottom_index<0 || bottom_index > nMaxSize)continue;
				  pBottomData[bottom_index] += pTopData[index];
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset1),(long)(pBottomData));
			dma_wait(&putreply,1);putreply=0;		
		}	
		
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
		if(use_top_mask>0)
			ldm_free(pTopMask,nMaskSize);
		else
			ldm_free(pMask,nMaskSize);
	}
}
void poolingBackwardAvg_f(SlavePoolingParam_f *pParam)
{
  const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,pool_size,nPoolIndex=0;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	
	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
			
	if((nTopSize+nBottomSize) > nMaxBuffSize)
	{
		nBottomSize = nMaxBuffSize - nTopSize;
		int nSplitCount=0,nSplitRows =0,nLeftRows = 0,nRows;
		int nTopSize1 = 0,nBottomIndex=0;
		nSplitRows = pooled_height_;
		int nKernelSize = kernel_h_*kernel_w_*sizeof(Type);
		while(nBottomSize < nKernelSize)
		{
			nSplitRows = nSplitRows>>1;			
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			nBottomSize = nMaxBuffSize - nTopSize;
			nSplitCount++;
		}
		if(nSplitCount <1){
			nSplitRows = 0;
			nLeftRows = pooled_height_;
			nTopSize1 = pooled_height_ * pooled_width_*sizeof(Type);
		}
	    else{
			nSplitCount = pooled_height_/nSplitRows;
			nLeftRows = pooled_height_%nSplitRows;
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);						
			nTopSize1 = nLeftRows*pooled_width_*sizeof(Type);							
		}
		nBottomSize = nMaxBuffSize - nTopSize;		
		pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);		
		
		for(i=0;i<nCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
						
			for(j=0;j<nSplitCount;j++)
			{
				dma_set_size(&pool_dmaget2, nTopSize);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows) * stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1)continue;
          nPoolIndex = ph*pooled_width_;
					nKernelSize = nRows*width_*sizeof(Type);
					
          nBottomIndex = nOffset1+hstart*width_;
					dma_set_size(&pool_dmaget2, nKernelSize);
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&getreply,1);getreply=0;
				
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						pool_index = nPoolIndex + pw;
						for ( h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for ( w = wstart; w < wend; ++w) {
							pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
						  }
						}
					}			  
					dma_set_size(&pool_dmaput2, nKernelSize);
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&putreply,1);putreply=0;
				}
			}
			if(nLeftRows > 0)
		    {
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaget2, nTopSize1);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows) * stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1)continue;
          nPoolIndex = ph*pooled_width_;
					nKernelSize = nRows*width_*sizeof(Type);
					
          nBottomIndex = nOffset1+hstart*width_;
					dma_set_size(&pool_dmaget2, nKernelSize);
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&getreply,1);getreply=0;
				
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						pool_index = nPoolIndex + pw;
						for ( h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for ( w = wstart; w < wend; ++w) {
							pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
						  }
						}
					}			  
					dma_set_size(&pool_dmaput2, nKernelSize);
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&putreply,1);putreply=0;
				}
			}			
		}	  
				
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;			
			for(j=0;j<nSplitCount;j++)
			{
				dma_set_size(&pool_dmaget2, nTopSize);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows) * stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1) continue;
          nPoolIndex = ph*pooled_width_;
					nKernelSize = nRows*width_*sizeof(Type);
					
          nBottomIndex = nOffset1+hstart*width_;
					dma_set_size(&pool_dmaget2, nKernelSize);
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&getreply,1);getreply=0;
				
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						pool_index = nPoolIndex + pw;
						for ( h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for ( w = wstart; w < wend; ++w) {
							pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
						  }
						}
					}			  
					dma_set_size(&pool_dmaput2, nKernelSize);
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&putreply,1);putreply=0;
				}
			}
			if(nLeftRows > 0)
		    {
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaget2, nTopSize1);  				
				dma(pool_dmaget2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&getreply,1);getreply=0;
				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows) * stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;
          if(nRows<1)continue;
					nPoolIndex = ph*pooled_width_;
					nKernelSize = nRows*width_*sizeof(Type);
					nBottomIndex = nOffset1+hstart*width_;
					dma_set_size(&pool_dmaget2, nKernelSize);
					dma(pool_dmaget2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&getreply,1);getreply=0;
				
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						pool_index = nPoolIndex + pw;
						for ( h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for ( w = wstart; w < wend; ++w) {
							pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
						  }
						}
					}			  
					dma_set_size(&pool_dmaput2, nKernelSize);
					dma(pool_dmaput2,(long)(pParam->pBottomData+nBottomIndex),(long)(pBottomData));	
					dma_wait(&putreply,1);putreply=0;
				}
			}	
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
	}
	else
	{ 
        pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);
		
		dma_set_size(&pool_dmaget2, nTopSize);
		dma_set_size(&pool_dmaput2, nBottomSize);
			
		for(i=0;i<nCount;i++)
		{
			nOffset = i*nMaxThreadsNum + myid;		
			dma(pool_dmaget2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));	
			memset(pBottomData,0,nBottomSize);
			dma_wait(&getreply,1);getreply=0;				
			
			for (ph = 0; ph < pooled_height_; ++ph) {
			    hstart = ph * stride_h_ - pad_h_;
				hend = min(hstart + kernel_h_, height_ + pad_h_);
				hstart = max(hstart, 0);
				hend = min(hend, height_);
				nPoolIndex = ph * pooled_width_;
				for (pw = 0; pw < pooled_width_; ++pw) {
					wstart = pw * stride_w_ - pad_w_;
					wend = min(wstart + kernel_w_, width_ + pad_w_);
					wstart = max(wstart, 0);
					wend = min(wend, width_);
					pool_size = (hend - hstart) * (wend - wstart);
					pool_index = nPoolIndex + pw;
					for ( h = hstart; h < hend; ++h) {
					  bottom_index = h * width_;
					  for ( w = wstart; w < wend; ++w) {
						pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
					  }
				    }
				}
			}
			
			dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));			
			dma_wait(&putreply,1);putreply=0;				
		}
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			dma(pool_dmaget2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));	
			memset(pBottomData,0,nBottomSize);
			dma_wait(&getreply,1);getreply=0;				
			for (ph = 0; ph < pooled_height_; ++ph) {
			    hstart = ph * stride_h_ - pad_h_;
				hend = min(hstart + kernel_h_, height_ + pad_h_);
				hstart = max(hstart, 0);
				hend = min(hend, height_);
				nPoolIndex = ph * pooled_width_;
				for (pw = 0; pw < pooled_width_; ++pw) {
					wstart = pw * stride_w_ - pad_w_;
					wend = min(wstart + kernel_w_, width_ + pad_w_);
					wstart = max(wstart, 0);
					wend = min(wend, width_);
					pool_size = (hend - hstart) * (wend - wstart);
					pool_index = nPoolIndex + pw;
					for ( h = hstart; h < hend; ++h) {
					  bottom_index = h * width_;
					  for ( w = wstart; w < wend; ++w) {
						pBottomData[bottom_index + w] += pTopData[pool_index] / pool_size;
					  }
				    }
				}
			}
			
			dma(pool_dmaput2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));			
			dma_wait(&putreply,1);putreply=0;			
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
	}
}
void poolingForwardMax_f(SlavePoolingParam_f *pParam)
{
    const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,nRows,nPoolIndex,nBottomIndex;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask;	
	int  *pMask;	
	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,dmaputmask,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	dma_set_op(&dmaputmask, DMA_PUT);
	dma_set_mode(&dmaputmask, PE_MODE);
	dma_set_reply(&dmaputmask, &putmaskreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
	if(use_top_mask>0)
	  nMaskSize = nTopSize;
    else
	  nMaskSize = pooled_height_ * pooled_width_*sizeof(int);  	
			
	if((nTopSize+nBottomSize+nMaskSize) > nMaxBuffSize)
	{
		nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
		int nSplitCount=0,nSplitRows =0,nLeftRows = 0;
		int nTopSize1 = 0,nMaskSize1=0;
		nSplitRows = pooled_height_;
		int nKernelSize = kernel_h_*width_*sizeof(Type);
		while(nBottomSize < nKernelSize)
		{
			nSplitRows = nSplitRows>>1;			
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			if(use_top_mask >0) nMaskSize = nTopSize;				
			else nMaskSize = nSplitRows*pooled_width_*sizeof(int);
			nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
			nSplitCount++;
		}
		if(nSplitCount <1){
			nSplitRows = 0;
			nLeftRows = pooled_height_;
			nTopSize1 = pooled_height_ * pooled_width_*sizeof(Type);
			nMaskSize1 = pooled_height_ * pooled_width_*sizeof(int);
		}
	  else{
			nSplitCount = pooled_height_/nSplitRows;
			nLeftRows = pooled_height_%nSplitRows;
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
	  	if(use_top_mask >0) 
			{
				nMaskSize = nTopSize;	
				nMaskSize1 = nLeftRows*pooled_width_*sizeof(Type);
			}
			else
			{
				nMaskSize = nSplitRows*pooled_width_*sizeof(int);
				nMaskSize1 = nLeftRows*pooled_width_*sizeof(int);
			}				
			nTopSize1 = nLeftRows*pooled_width_*sizeof(Type);							
		}
		nBottomSize = nMaxBuffSize - nTopSize - nMaskSize;
		if(use_top_mask>0)
		{
			nMaskSize = nTopSize;
			pTopMask  = (Type*)(long)ldm_malloc(nMaskSize);
		}
		else
		{
			pMask  = (int*)(long)ldm_malloc(nMaskSize);	
        }	
		
		pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);		
				
		for(i=0;i<nCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
						
			for(j=0;j<nSplitCount;j++)
			{	
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);	
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
          if(nRows <1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					nOffset = hstart*width_;  
					dma_wait(&getreply,1);getreply=0;	
					for (pw = 0; pw < pooled_width_; ++pw) 
					{
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_);
						wstart = max(wstart, 0);
						pool_index = nPoolIndex+pw;
						pTopData[pool_index] = -FLT_MAX;
						
						for (h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for (w = wstart; w < wend; ++w) {
							index = bottom_index + w;
							if (pBottomData[index] > pTopData[pool_index]) {
							  pTopData[pool_index] = pBottomData[index];						  
							  if(use_top_mask>0) 
								pTopMask[pool_index] = index+nOffset;  
							  else
								pMask[pool_index] = index+nOffset;  
							}
						  }
						}					
					}					
				}
				nOffset = j * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaput2,nTopSize);				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_set_size(&dmaputmask,nMaskSize);				
				if(use_top_mask>0) 
					dma(dmaputmask,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
				else
					dma(dmaputmask,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
				dma_wait(&putreply,1);putreply=0;				
				dma_wait(&putmaskreply,1);putmaskreply=0;
			}
			if(nLeftRows > 0)
		    {				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);	
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
          if(nRows<1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					nOffset = hstart*width_;  
					dma_wait(&getreply,1);getreply=0;	
					for (pw = 0; pw < pooled_width_; ++pw) 
					{
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_);
						wstart = max(wstart, 0);
						pool_index = nPoolIndex+pw;
						pTopData[pool_index] = -FLT_MAX;
						
						for (h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for (w = wstart; w < wend; ++w) {
							index = bottom_index + w;
							if (pBottomData[index] > pTopData[pool_index]) {
							  pTopData[pool_index] = pBottomData[index];						  
							  if(use_top_mask>0) 
								pTopMask[pool_index] = index+nOffset;  
							  else
								pMask[pool_index] = index+nOffset;  
							}
						  }
					}					
					}					
				}
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaput2,nTopSize1);				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_set_size(&dmaputmask,nMaskSize1);				
				if(use_top_mask>0) 
					dma(dmaputmask,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
				else
					dma(dmaputmask,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
				dma_wait(&putreply,1);putreply=0;				
				dma_wait(&putmaskreply,1);putmaskreply=0;
			}
		}	  
				
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;			
			for(j=0;j<nSplitCount;j++)
			{	
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);	
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
          if(nRows<1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					nOffset = hstart*width_;  
					dma_wait(&getreply,1);getreply=0;	
					for (pw = 0; pw < pooled_width_; ++pw) 
					{
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_);
						wstart = max(wstart, 0);
						pool_index = nPoolIndex+pw;
						pTopData[pool_index] = -FLT_MAX;
						
						for (h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for (w = wstart; w < wend; ++w) {
							index = bottom_index + w;
							if (pBottomData[index] > pTopData[pool_index]) {
							  pTopData[pool_index] = pBottomData[index];						  
							  if(use_top_mask>0) 
								pTopMask[pool_index] = index+nOffset;  
							  else
								pMask[pool_index] = index+nOffset;  
							}
						  }
						}					
					}					
				}
				nOffset = j * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaput2,nTopSize);				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_set_size(&dmaputmask,nMaskSize);				
				if(use_top_mask>0) 
					dma(dmaputmask,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
				else
					dma(dmaputmask,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
				dma_wait(&putreply,1);putreply=0;				
				dma_wait(&putmaskreply,1);putmaskreply=0;
			}
			if(nLeftRows > 0)
		    {				
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);	
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
          if(nRows<1) continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					nOffset = hstart*width_;  
					dma_wait(&getreply,1);getreply=0;	
					for (pw = 0; pw < pooled_width_; ++pw) 
					{
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_);
						wstart = max(wstart, 0);
						pool_index = nPoolIndex+pw;
						pTopData[pool_index] = -FLT_MAX;
						
						for (h = 0; h < nRows; ++h) {
						  bottom_index = h * width_;
						  for (w = wstart; w < wend; ++w) {
							index = bottom_index + w;
							if (pBottomData[index] > pTopData[pool_index]) {
							  pTopData[pool_index] = pBottomData[index];						  
							  if(use_top_mask>0) 
								pTopMask[pool_index] = index+nOffset;  
							  else
								pMask[pool_index] = index+nOffset;  
							}
						  }
						}					
					}					
				}
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				dma_set_size(&pool_dmaput2,nTopSize1);				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_set_size(&dmaputmask,nMaskSize1);				
				if(use_top_mask>0) 
					dma(dmaputmask,(long)(pParam->pTopMask+nOffset0+nOffset),(long)(pTopMask));
				else
					dma(dmaputmask,(long)(pParam->pMask+nOffset0+nOffset),(long)(pMask));
				dma_wait(&putreply,1);putreply=0;				
				dma_wait(&putmaskreply,1);putmaskreply=0;
			}
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
		if(use_top_mask>0)
			ldm_free(pTopMask,nMaskSize);
		else
			ldm_free(pMask,nMaskSize);
	}
	else
	{ 
        pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);
			
		if(use_top_mask>0)
		  pTopMask  = (Type*)(long)ldm_malloc(nMaskSize);
		else
		  pMask  = (int*)(long)ldm_malloc(nMaskSize); 
		
		dma_set_size(&pool_dmaput2, nTopSize);
		dma_set_size(&pool_dmaget2, nBottomSize);
	    dma_set_size(&dmaputmask, nMaskSize);
			
		for(i=0;i<nCount;i++)
		{
			nOffset = i*nMaxThreadsNum + myid;		
			dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
			dma_wait(&getreply,1);getreply=0;				
			for (ph = 0; ph < pooled_height_; ++ph) {
			  hstart = ph * stride_h_ - pad_h_;
			  hend = min(hstart + kernel_h_, height_);
			  hstart = max(hstart, 0);
		    nPoolIndex = ph*pooled_width_;				
			  for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_);
				wstart = max(wstart, 0);
				pool_index = nPoolIndex + pw;
				pTopData[pool_index] = -FLT_MAX;
					
				for (h = hstart; h < hend; ++h) {
				  nBottomIndex = h * width_;
				  for (w = wstart; w < wend; ++w) {
					index = nBottomIndex + w;
					if (pBottomData[index] > pTopData[pool_index]) {
					  pTopData[pool_index] = pBottomData[index];						  
					  if(use_top_mask>0) 
		  				pTopMask[pool_index] = index;  
					  else
			  			pMask[pool_index] = index;
					}
				  }
				}
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));
			if(use_top_mask>0) 
				dma(dmaputmask,(long)(pParam->pTopMask+nOffset*nTopOffset),(long)(pTopMask));
			else
				dma(dmaputmask,(long)(pParam->pMask+nOffset*nTopOffset),(long)(pMask));
			
			dma_wait(&putreply,1);putreply=0;				
			dma_wait(&putmaskreply,1);putmaskreply=0;	
	    }
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
			dma_wait(&getreply,1);getreply=0;				
			for (ph = 0; ph < pooled_height_; ++ph) {
			  hstart = ph * stride_h_ - pad_h_;
			  hend = min(hstart + kernel_h_, height_);
			  hstart = max(hstart, 0);
		    nPoolIndex = ph*pooled_width_;				
			  for (pw = 0; pw < pooled_width_; ++pw) {
				wstart = pw * stride_w_ - pad_w_;
				wend = min(wstart + kernel_w_, width_);
				wstart = max(wstart, 0);
				pool_index = nPoolIndex + pw;
				pTopData[pool_index] = -FLT_MAX;
					
				for (h = hstart; h < hend; ++h) {
				  nBottomIndex = h * width_;
				  for (w = wstart; w < wend; ++w) {
					index = nBottomIndex + w;
					if (pBottomData[index] > pTopData[pool_index]) {
					  pTopData[pool_index] = pBottomData[index];						  
					  if(use_top_mask>0) 
						pTopMask[pool_index] = index;  
					  else
						pMask[pool_index] = index;
					}
				  }
				}
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));
			if(use_top_mask>0) 
				dma(dmaputmask,(long)(pParam->pTopMask+nOffset*nTopOffset),(long)(pTopMask));
			else
				dma(dmaputmask,(long)(pParam->pMask+nOffset*nTopOffset),(long)(pMask));
			
			dma_wait(&putreply,1);putreply=0;				
			dma_wait(&putmaskreply,1);putmaskreply=0;		
		}	
		
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
		if(use_top_mask>0)
			ldm_free(pTopMask,nMaskSize);
		else
			ldm_free(pMask,nMaskSize);
	}
}
void poolingForwardAvg_f(SlavePoolingParam_f *pParam)
{
    const int nMaxBuffSize = 49152;//58KB 
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nOffset0,nOffset1;
	int nBottomOffset,nTopOffset,use_top_mask,nRows,nPoolIndex,pool_size;
	int ph,pw,hstart,hend,wstart,wend,pool_index,h,w,index,bottom_index;
	Type *pTopData,*pBottomData,*pTopMask,dSum=0;	
	int  *pMask;	
	
	volatile int getreply=0,putreply=0,putmaskreply=0;	
	int myid = athread_get_id(-1);
	
	pooled_height_ = pParam->pooled_height_;
	pooled_width_  = pParam->pooled_width_;
	stride_h_ = pParam->stride_h_;
	stride_w_ = pParam->stride_w_;
	pad_h_ = pParam->pad_h_;
	pad_w_ = pParam->pad_w_;
	kernel_h_ = pParam->kernel_h_;
	kernel_w_ = pParam->kernel_w_;
	height_ = pParam->height_;
	width_  = pParam->width_;	
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nThreadsNum;
	nLeftMaxThreadsNum = pParam->nLeftThreadsNum;
	nBottomOffset = pParam->nBottomOffset;
	nTopOffset = pParam->nTopOffset;
	use_top_mask = pParam->use_top_mask;
	
	if(myid >= nMaxThreadsNum) return;	
	//dma_desc pool_dmaget2,pool_dmaput2;
	dma_set_op(&pool_dmaget2, DMA_GET);
	dma_set_mode(&pool_dmaget2, PE_MODE);
	dma_set_reply(&pool_dmaget2, &getreply);
	
	dma_set_op(&pool_dmaput2, DMA_PUT);
	dma_set_mode(&pool_dmaput2, PE_MODE);
	dma_set_reply(&pool_dmaput2, &putreply);	
	
	int nTopSize = pooled_height_ * pooled_width_*sizeof(Type),i=0,j=0;
	int nBottomSize = height_ * width_*sizeof(Type),nMaskSize=0;
			
	if((nTopSize+nBottomSize) > nMaxBuffSize)
	{
		nBottomSize = nMaxBuffSize - nTopSize;
		int nSplitCount=0,nSplitRows =0,nLeftRows = 0;
		int nTopSize1 = 0,nBottomIndex;
		nSplitRows = pooled_height_;
		int nKernelSize = kernel_h_*width_*sizeof(Type);
		while(nBottomSize < nKernelSize)
		{
			nSplitRows = nSplitRows>>1;			
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);
			nBottomSize = nMaxBuffSize - nTopSize;
			nSplitCount++;
		}
		if(nSplitCount <1){
			nSplitRows = 0;
			nLeftRows = pooled_height_;
			nTopSize1 = pooled_height_ * pooled_width_*sizeof(Type);
		}
	  else{
			nSplitCount = pooled_height_/nSplitRows;
			nLeftRows = pooled_height_%nSplitRows;
			nTopSize = nSplitRows*pooled_width_*sizeof(Type);						
			nTopSize1 = nLeftRows*pooled_width_*sizeof(Type);							
		}
		nBottomSize = nMaxBuffSize - nTopSize;		
		pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);		
		
		for(i=0;i<nCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;
						
			for(j=0;j<nSplitCount;j++)
			{
				memset(pTopData,0,nTopSize);
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1) continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					dma_wait(&getreply,1);getreply=0;				
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						dSum = 0;
						for (h = 0; h < nRows; ++h) {
						  for ( w = wstart; w < wend; ++w) {
							dSum +=	pBottomData[h * width_ + w];
						  }
						}
						pTopData[nPoolIndex + pw] = dSum/pool_size;				
					}					
				}
				dma_set_size(&pool_dmaput2, nTopSize);  				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&putreply,1);putreply=0;
			}
			if(nLeftRows > 0)
		    {
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				memset(pTopData,0,nTopSize);
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					dma_wait(&getreply,1);getreply=0;				
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						dSum = 0;
						for (h = 0; h < nRows; ++h) {
						  for ( w = wstart; w < wend; ++w) {
							dSum +=	pBottomData[h * width_ + w];
						  }
						}
						pTopData[nPoolIndex + pw] = dSum/pool_size;				
					}					
				}
				dma_set_size(&pool_dmaput2, nTopSize1);  				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&putreply,1);putreply=0;
			}			
		}	  
				
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			nOffset0 = nOffset * nTopOffset;
			nOffset1 = nOffset * nBottomOffset;			
			for(j=0;j<nSplitCount;j++)
			{
				memset(pTopData,0,nTopSize);
				for (ph = 0; ph < nSplitRows; ++ph) 
				{
					hstart = (ph+j*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					dma_wait(&getreply,1);getreply=0;				
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						dSum = 0;
						for (h = 0; h < nRows; ++h) {
						  for ( w = wstart; w < wend; ++w) {
							dSum +=	pBottomData[h * width_ + w];
						  }
						}
						pTopData[nPoolIndex + pw] = dSum/pool_size;				
					}					
				}
				dma_set_size(&pool_dmaput2, nTopSize);  				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+j*nSplitRows*pooled_width_),(long)(pTopData));
				dma_wait(&putreply,1);putreply=0;
			}
			if(nLeftRows > 0)
		    {
				nOffset = nSplitCount * nSplitRows*pooled_width_;
				memset(pTopData,0,nTopSize);
				for (ph = 0; ph < nLeftRows; ++ph) 
				{
					hstart = (ph+nSplitCount*nSplitRows)* stride_h_ - pad_h_;
					hend = min(hstart + kernel_h_, height_);				
					hstart = max(hstart, 0);				
					nRows = hend - hstart;				
					if(nRows<1)continue;
					dma_set_size(&pool_dmaget2,nRows*width_ *sizeof(Type));				
					dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset1+hstart*width_),(long)(pBottomData));
					nPoolIndex = ph*pooled_width_;
					dma_wait(&getreply,1);getreply=0;				
					
					for (pw = 0; pw < pooled_width_; ++pw) {
						wstart = pw * stride_w_ - pad_w_;
						wend = min(wstart + kernel_w_, width_ + pad_w_);
						wstart = max(wstart, 0);
						wend = min(wend, width_);
						pool_size = (hend - hstart) * (wend - wstart);
						dSum = 0;
						for (h = 0; h < nRows; ++h) {
						  for ( w = wstart; w < wend; ++w) {
							dSum +=	pBottomData[h * width_ + w];
						  }
						}
						pTopData[nPoolIndex + pw] = dSum/pool_size;				
					}					
				}
				dma_set_size(&pool_dmaput2, nTopSize1);  				
				dma(pool_dmaput2,(long)(pParam->pTopData+nOffset0+nOffset),(long)(pTopData));
				dma_wait(&putreply,1);putreply=0;
			}
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
	}
	else
	{ 
    pTopData  = (Type*)(long)ldm_malloc(nTopSize);
		pBottomData = (Type*)(long)ldm_malloc(nBottomSize);
		
		dma_set_size(&pool_dmaget2, nBottomSize);
		dma_set_size(&pool_dmaput2, nTopSize);
		
		for(i=0;i<nCount;i++)
		{
			nOffset = i*nMaxThreadsNum + myid;		
			dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
			memset(pTopData,0,nTopSize);
			dma_wait(&getreply,1);getreply=0;				
			
			for (ph = 0; ph < pooled_height_; ++ph) {
			    hstart = ph * stride_h_ - pad_h_;
				hend = min(hstart + kernel_h_, height_ + pad_h_);
				hstart = max(hstart, 0);
				hend = min(hend, height_);
				nPoolIndex = ph*pooled_width_;
				for (pw = 0; pw < pooled_width_; ++pw) {
					wstart = pw * stride_w_ - pad_w_;
					wend = min(wstart + kernel_w_, width_ + pad_w_);
					wstart = max(wstart, 0);
					wend = min(wend, width_);
					pool_size = (hend - hstart) * (wend - wstart);
					dSum = 0;
					for (h = hstart; h < hend; ++h) {
					  for ( w = wstart; w < wend; ++w) {
						dSum +=	pBottomData[h * width_ + w];
					  }
				    }
				    pTopData[nPoolIndex + pw] = dSum/pool_size;				
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));			
			dma_wait(&putreply,1);putreply=0;				
		}
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nCount*nMaxThreadsNum + myid;
			dma(pool_dmaget2,(long)(pParam->pBottomData+nOffset*nBottomOffset),(long)(pBottomData));
			memset(pTopData,0,nTopSize);
			dma_wait(&getreply,1);getreply=0;				
			
			for (ph = 0; ph < pooled_height_; ++ph) {
			    hstart = ph * stride_h_ - pad_h_;
				hend = min(hstart + kernel_h_, height_ + pad_h_);
				hstart = max(hstart, 0);
				hend = min(hend, height_);
				nPoolIndex = ph*pooled_width_;
				for (pw = 0; pw < pooled_width_; ++pw) {
					wstart = pw * stride_w_ - pad_w_;
					wend = min(wstart + kernel_w_, width_ + pad_w_);
					wstart = max(wstart, 0);
					wend = min(wend, width_);
					pool_size = (hend - hstart) * (wend - wstart);
					dSum = 0;
					for ( h = hstart; h < hend; ++h) {
					  for ( w = wstart; w < wend; ++w) {
						dSum +=	pBottomData[h * width_ + w];
					  }
				    }
				    pTopData[nPoolIndex + pw] = dSum/pool_size;				
			  }
			}
			dma(pool_dmaput2,(long)(pParam->pTopData+nOffset*nTopOffset),(long)(pTopData));			
			dma_wait(&putreply,1);putreply=0;			
		}
		ldm_free(pTopData,nTopSize);
		ldm_free(pBottomData,nBottomSize);
	}
}
