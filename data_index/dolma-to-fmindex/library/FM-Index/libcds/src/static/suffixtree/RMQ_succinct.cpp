/* Copyright (C) 2010, Rodrigo Cánovas, all rights reserved.
 *
 *This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include <RMQ_succinct.h>

namespace cds_static{
	const unsigned int RMQ_succinct::Catalan[17][17] = {
		{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
		{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
		{0,0,2,5,9,14,20,27,35,44,54,65,77,90,104,119,135},
		{0,0,0,5,14,28,48,75,110,154,208,273,350,440,544,663,798},
		{0,0,0,0,14,42,90,165,275,429,637,910,1260,1700,2244,2907,3705},
		{0,0,0,0,0,42,132,297,572,1001,1638,2548,3808,5508,7752,10659,14364},
		{0,0,0,0,0,0,132,429,1001,2002,3640,6188,9996,15504,23256,33915,48279},
		{0,0,0,0,0,0,0,429,1430,3432,7072,13260,23256,38760,62016,95931,144210},
		{0,0,0,0,0,0,0,0,1430,4862,11934,25194,48450,87210,149226,245157,389367},
		{0,0,0,0,0,0,0,0,0,4862,16796,41990,90440,177650,326876,572033,961400},
		{0,0,0,0,0,0,0,0,0,0,16796,58786,149226,326876,653752,1225785,2187185},
		{0,0,0,0,0,0,0,0,0,0,0,58786,208012,534888,1188640,2414425,4601610},
		{0,0,0,0,0,0,0,0,0,0,0,0,208012,742900,1931540,4345965,8947575},
		{0,0,0,0,0,0,0,0,0,0,0,0,0,742900,2674440,7020405,15967980},
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,2674440,9694845,25662825},
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9694845,35357670},
		{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35357670}
	};

	const int RMQ_succinct::minus_infinity = INT_MIN;

	const char RMQ_succinct::LSBTable256[256] = {
		0,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
		4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	};

	unsigned int RMQ_succinct::lsb(DTsucc v) {
		return LSBTable256[v];
	}

	const char RMQ_succinct::LogTable256[256] = {
		0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
		4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
		5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
		5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
		7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
	};

	RMQ_succinct::RMQ_succinct(){
		a = NULL;
		n = 0;
		M = NULL;
		M_depth = 0;
		Mprime = NULL;
		Mprime_depth = 0;
		type = NULL;
		Prec = NULL;
		s = 0;
		sprime = 0;
		sprimeprime = 0;
		nb = 0;
		nsb = 0;
		nmb = 0;
	}

	unsigned int RMQ_succinct::log2fast(unsigned int v) {
		unsigned int c = 0;          // c will be lg(v)
		register unsigned int t, tt; // temporaries

		if ((tt = v >> 16))
			c = (t = v >> 24) ? 24 + LogTable256[t] : 16 + LogTable256[tt & 0xFF];
		else 
			c = (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
		return c;
	}

//	const DTsucc RMQ_succinct::HighestBitsSet[8] = {~0, ~1, ~3, ~7, ~15, ~31, ~63, ~127};
    const DTsucc RMQ_succinct::HighestBitsSet[8] = {static_cast<DTsucc>(~0), static_cast<DTsucc>(~1), static_cast<DTsucc>(~3), static_cast<DTsucc>(~7), static_cast<DTsucc>(~15), static_cast<DTsucc>(~31), static_cast<DTsucc>(~63), static_cast<DTsucc>(~127)};

	DTsucc RMQ_succinct::clearbits(DTsucc n, unsigned int x) {
		return n & HighestBitsSet[x];
	}

	unsigned int RMQ_succinct::query(unsigned int i, unsigned int j) {
		unsigned int mb_i = microblock(i);     // i's microblock
		unsigned int mb_j = microblock(j);     // j's microblock
		unsigned int min, min_i, min_j; 		// min: to be returned
		unsigned int s_mi = mb_i * s;           // start of i's microblock
		unsigned int i_pos = i - s_mi;          // pos. of i in its microblock

		if (mb_i == mb_j) { // only one microblock-query
			min_i = clearbits(Prec[type[mb_i]][j-s_mi], i_pos);
			min = min_i == 0 ? j : s_mi + lsb(min_i);
		}
		else {
			unsigned int b_i = block(i);      // i's block
			unsigned int b_j = block(j);      // j's block
			unsigned int s_mj = mb_j * s;     // start of j's microblock
			unsigned int j_pos = j - s_mj;    // position of j in its microblock
			min_i = clearbits(Prec[type[mb_i]][s-1], i_pos);
			min = min_i == 0 ? s_mi + s - 1 : s_mi + lsb(min_i); // left in-microblock-query
			min_j = Prec[type[mb_j]][j_pos] == 0 ?
				j : s_mj + lsb(Prec[type[mb_j]][j_pos]);         // right in-microblock-query
			if (a[min_j] < a[min]) min = min_j;

			if (mb_j > mb_i + 1) { // otherwise we're done!
				unsigned int s_bi = b_i * sprime;      // start of block i
				unsigned int s_bj = b_j * sprime;      // start of block j
				if (s_bi+s > i) { // do another microblock-query!
					mb_i++; // go one microblock to the right
					min_i = Prec[type[mb_i]][s-1] == 0 ?
						s_bi + sprime - 1 : s_mi + s + lsb(Prec[type[mb_i]][s-1]); // right in-block-query
					if (a[min_i] < a[min]) min = min_i;
				}
				if (j >= s_bj+s) { // and yet another microblock-query!
					mb_j--; // go one microblock to the left
					min_j = Prec[type[mb_j]][s-1] == 0 ?
						s_mj - 1 : s_bj + lsb(Prec[type[mb_j]][s-1]); // right in-block-query
					if (a[min_j] < a[min]) min = min_j;
				}

				unsigned int block_difference = b_j - b_i;
				if (block_difference > 1) { // otherwise we're done!
					unsigned int k, twotothek, block_tmp;  // for index calculations in M and M'
					b_i++; // block where out-of-block-query starts
					if (s_bj - s_bi - sprime <= sprimeprime) { // just one out-of-block-query
						k = log2fast(block_difference - 2);
						twotothek = 1 << k; // 2^k
						i = m(k, b_i); j = m(k, b_j-twotothek);
						min_i = a[i] <= a[j] ? i : j;
					}
					else { // here we have to answer a superblock-query:
						unsigned int sb_i = superblock(i); // i's superblock
						unsigned int sb_j = superblock(j); // j's superblock

						block_tmp = block((sb_i+1)*sprimeprime); // end of left out-of-block-query
						k = log2fast(block_tmp - b_i);
						twotothek = 1 << k; // 2^k
						i = m(k, b_i); j = m(k, block_tmp+1-twotothek);
						min_i = a[i] <= a[j] ? i : j;

						block_tmp = block(sb_j*sprimeprime); // start of right out-of-block-query
						k = log2fast(b_j - block_tmp);
						twotothek = 1 << k; // 2^k
						block_tmp--; // going one block to the left doesn't harm and saves some tests
						i = m(k, block_tmp); j = m(k, b_j-twotothek);
						min_j = a[i] <= a[j] ? i : j;

						if (a[min_j] < a[min_i]) min_i = min_j;

						if (sb_j > sb_i + 1) { // finally, the superblock-query:
							k = log2fast(sb_j - sb_i - 2);
							twotothek = 1 << k;
							i = Mprime[k][sb_i+1]; j = Mprime[k][sb_j-twotothek];
							min_j = a[i] <= a[j] ? i : j;
							if (a[min_j] < a[min_i]) min_i = min_j; // does NOT always return leftmost min!!!
						}
					}
					if (a[min_i] < a[min]) min = min_i; // does NOT always return leftmost min!!!
				}
			}
		}
		return min;
	}

	/**
	 * Standard Constructor. a is the array to be prepared for RMQ.
	 * n is the size of the array.
	 * */
	RMQ_succinct::RMQ_succinct(int* a, unsigned int n) {
		this->a = a;
		this->n = n;
		s = 1 << 3;	             // microblock-size
		sprime = 1 << 4;         // block-size
		sprimeprime = 1 << 8;	 // superblock-size
		nb = block(n-1)+1;       // number of blocks
		nsb = superblock(n-1)+1; // number of superblocks
		nmb = microblock(n-1)+1; // number of microblocks

		// The following is necessary because we've fixed s, s' and s'' according to the computer's
		// word size and NOT according to the input size. This may cause the (super-)block-size
		// to be too big, or, in other words, the array too small. If this code is compiled on
		// a 32-bit computer, this happens iff n < 113. For such small instances it isn't 
		// advisable anyway to use this data structure, because simpler methods are faster and 
		// less space consuming.
		if (nb<sprimeprime/(2*sprime)) { cerr << "Array too small...exit\n"; exit(-1); }
		// Type-calculation for the microblocks and pre-computation of in-microblock-queries:
		type = new DTsucc2[nmb];
		Prec = new DTsucc*[Catalan[s][s]];
		for (unsigned int i = 0; i < Catalan[s][s]; i++) {
			Prec[i] = new DTsucc[s];
			for(unsigned int j=0; j<s;j++)
				Prec[i][j]=0;
			Prec[i][0] = 1; // init with impossible value
		}

		int* rp = new int[s+1];   // rp: rightmost path in Cart. tree
		unsigned int z = 0;            // index in array a
		unsigned int start;            // start of current block
		unsigned int end;              // end of current block
		unsigned int q;                // position in Catalan triangle
		unsigned int p;                // --------- " ----------------
		rp[0] = minus_infinity; // stopper (minus infinity)

		// prec[i]: the jth bit is 1 iff j is 1. pos. to the left of i where a[j] < a[i] 
		unsigned int* gstack = new unsigned int[s];
		unsigned int gstacksize;
		unsigned int g; // first position to the left of i where a[g[i]] < a[i]

		for (unsigned int i = 0; i < nmb; i++) { // step through microblocks
			start = z;            // init start
			end = start + s;      // end of block (not inclusive!)
			if (end > n) end = n; // last block could be smaller than s!
			// compute block type as in Fischer/Heun CPM'06:
			q = s;        // init q
			p = s-1;      // init p
			type[i] = 0;  // init type (will be increased!)
			rp[1] = a[z]; // init rightmost path

			while (++z < end) {   // step through current block:
				p--;
				while (rp[q-p-1] > a[z]) {
					type[i] += Catalan[p][q]; // update type
					q--;
				}
				rp[q-p] = a[z]; // add last element to rightmost path
			}

			// precompute in-block-queries for this microblock (if necessary)
			// as in Alstrup et al. SPAA'02:
			if (Prec[type[i]][0] == 1) {
				Prec[type[i]][0] = 0;
				gstacksize = 0;
				for (unsigned int j = start; j < end; j++) {
					while(gstacksize > 0 && (a[j] < a[gstack[gstacksize-1]])) {
						gstacksize--;
					}
					if(gstacksize > 0) {
						g = gstack[gstacksize-1];
						Prec[type[i]][j-start] = Prec[type[i]][g-start] | (1 << (g % s));
					}
					else Prec[type[i]][j-start] = 0;
					gstack[gstacksize++] = j;
				}
			}
		}
		delete[] rp;
		delete[] gstack;

		// space for out-of-block- and out-of-superblock-queries:
		M_depth = (unsigned int) floor(log2(((double) sprimeprime / (double) sprime)));
		M = new DTsucc*[M_depth];
		M[0] = new DTsucc[nb];
		Mprime_depth = (unsigned int) floor(log2(nsb)) + 1;
		Mprime = new unsigned int*[Mprime_depth];
		Mprime[0] = new unsigned int[nsb];

		// fill 0'th rows of M and Mprime:
		z = 0; // minimum in current block
		q = 0; // pos. of min in current superblock
		g = 0; // number of current superblock
		for (unsigned int i = 0; i < nb; i++) { // step through blocks
			start = z;              // init start
			p = start;              // init minimum
			end = start + sprime;   // end of block (not inclusive!)
			if (end > n) end = n;   // last block could be smaller than sprime!
			if (a[z] < a[q]) q = z; // update minimum in superblock

			while (++z < end) { // step through current block:
				if (a[z] < a[p]) p = z; // update minimum in block
				if (a[z] < a[q]) q = z; // update minimum in superblock
			}
			M[0][i] = p-start;                     // store index of block-minimum (offset!)
			if (z % sprimeprime == 0 || z == n) {  // reached end of superblock?
				Mprime[0][g++] = q;               // store index of superblock-minimum
				q = z;
			}
		}

		// fill M
		unsigned int dist = 1; // always 2^(j-1)
		for (unsigned int j = 1; j < M_depth; j++) {
			M[j] = new DTsucc[nb];
			for (unsigned int i = 0; i < nb - dist; i++) { // be careful: loop may go too far
				M[j][i] = a[m(j-1, i)] <= a[m(j-1,i+dist)] ?
					M[j-1][i] : M[j-1][i+dist] + (dist*sprime); // add 'skipped' elements in a
			}
			for (unsigned int i = nb - dist; i < nb; i++) M[j][i] = M[j-1][i]; // fill overhang
			dist *= 2;
		}
		
		// fill M':
		dist = 1; // always 2^(j-1)
		for (unsigned int j = 1; j < Mprime_depth; j++) {
			Mprime[j] = new unsigned int[nsb];
			for (unsigned int i = 0; i < nsb - dist; i++) {
				Mprime[j][i] = a[Mprime[j-1][i]] <= a[Mprime[j-1][i+dist]] ?
					Mprime[j-1][i] : Mprime[j-1][i+dist];
			}
			for (unsigned int i = nsb - dist; i < nsb; i++) Mprime[j][i] = Mprime[j-1][i]; // overhang
			dist *= 2;
		}
	}

	uint RMQ_succinct::getSize(){
		uint mem = 0;
		mem += sizeof(RMQ_succinct);
		mem += sizeof(int)*n;
		mem += sizeof(DTsucc2)*nmb;
		mem += sizeof(DTsucc)*s;
		mem += sizeof(DTsucc)*nb;
		mem += sizeof(unsigned int)*nsb;
		mem += sizeof(DTsucc)*nb;
		mem += sizeof(unsigned int)*nsb;
		return mem;
	}

	void RMQ_succinct::save(ofstream & fp){
		saveValue(fp,n);
		saveValue(fp, a, n);
		saveValue(fp, type, nmb);
		for(uint i=0; i < M_depth; i++)
			saveValue(fp, M[i], nb);
		for(uint i=0; i < Mprime_depth; i++)
			saveValue(fp, Mprime[i], nsb);
		for(uint i=0; i < Catalan[s][s]; i++)
			saveValue(fp, Prec[i], s);
	}

	RMQ_succinct * RMQ_succinct::load(ifstream & fp){
		RMQ_succinct *rmq = new RMQ_succinct();
		rmq->s = 1<<3;
		rmq->sprime = 1<<4;
		rmq->sprimeprime = 1<<8;
		rmq->n = loadValue<uint>(fp);
		rmq->nb = rmq->block(rmq->n-1)+1;
		rmq->nsb = rmq->superblock(rmq->n-1)+1;
		rmq->nmb = rmq->microblock(rmq->n-1)+1;
		rmq->M_depth = (uint) floor(log2(((double) rmq->sprimeprime / (double) rmq->sprime)));
		rmq->Mprime_depth = (uint) floor(log2(rmq->nsb)) + 1;
		rmq->a = loadValue<int>(fp, rmq->n);
		rmq->type =  loadValue<DTsucc2>(fp, rmq->nmb);
		rmq->M = new DTsucc*[rmq->M_depth];
		for(uint i=0; i < rmq->M_depth; i++)
			rmq->M[i] = loadValue<DTsucc>(fp, rmq->nb);
		rmq->Mprime = new uint*[rmq->Mprime_depth];
		for(uint i=0; i < rmq->Mprime_depth; i++)
			rmq->Mprime[i] = loadValue<uint>(fp, rmq->nsb);
		rmq->Prec = new DTsucc*[Catalan[rmq->s][rmq->s]];
		for(uint i=0; i < Catalan[rmq->s][rmq->s]; i++)
			rmq->Prec[i] = loadValue<DTsucc>(fp, rmq->s);
		return rmq;
	}

	/**
	 * Destructor. Deletes allocated space.
	 **/
	RMQ_succinct::~RMQ_succinct() {
		delete[] type;
		for (unsigned int i = 0; i < Catalan[s][s]; i++) 
			delete[] Prec[i];
		delete[] Prec;
		for (unsigned int i = 0; i < M_depth; i++) 
			delete[] M[i];
		delete[] M;
		for (unsigned int i = 0; i < Mprime_depth; i++) 
			delete[] Mprime[i];
		delete[] Mprime;
		delete[] a;
	}

};

