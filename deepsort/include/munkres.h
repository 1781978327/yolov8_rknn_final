/*
 *   Copyright (c) 2007 John Weaver
 *   Copyright (c) 2015 Miroslav Krajicek
 */

#ifndef _MUNKRES_H_
#define _MUNKRES_H_

#include "matrix.h"

#include <list>
#include <utility>
#include <iostream>
#include <cmath>
#include <limits>

template<typename Data> class Munkres
{
    static constexpr int NORMAL = 0;
    static constexpr int STAR   = 1;
    static constexpr int PRIME  = 2;
public:
    void solve(Matrix<Data> &m) {
        const size_t rows = m.rows(),
                columns = m.columns(),
                size = XYZMAX(rows, columns);
        this->matrix = m;
        if ( rows != columns ) {
            matrix.resize(size, size, matrix.mmax());
        }
        mask_matrix.resize(size, size);
        row_mask = new bool[size];
        col_mask = new bool[size];
        for ( size_t i = 0 ; i < size ; i++ ) {
            row_mask[i] = false;
            col_mask[i] = false;
        }
        replace_infinites(matrix);
        minimize_along_direction(matrix, rows >= columns);
        minimize_along_direction(matrix, rows <  columns);
        int step = 1;
        while ( step ) {
            switch ( step ) {
            case 1: step = step1(); break;
            case 2: step = step2(); break;
            case 3: step = step3(); break;
            case 4: step = step4(); break;
            case 5: step = step5(); break;
            }
        }
        for ( size_t row = 0 ; row < size ; row++ ) {
            for ( size_t col = 0 ; col < size ; col++ ) {
                if ( mask_matrix(row, col) == STAR ) {
                    matrix(row, col) = 0;
                } else {
                    matrix(row, col) = -1;
                }
            }
        }
        matrix.resize(rows, columns);
        m = matrix;
        delete [] row_mask;
        delete [] col_mask;
    }
    static void replace_infinites(Matrix<Data> &matrix) {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        double max = matrix(0, 0);
        constexpr auto infinity = std::numeric_limits<double>::infinity();
        for ( size_t row = 0 ; row < rows ; row++ ) {
            for ( size_t col = 0 ; col < columns ; col++ ) {
                if ( matrix(row, col) != infinity ) {
                    if ( max == infinity ) {
                        max = matrix(row, col);
                    } else {
                        max = XYZMAX(max, matrix(row, col));
                    }
                }
            }
        }
        if ( max == infinity ) {
            max = 0;
        } else {
            max++;
        }
        for ( size_t row = 0 ; row < rows ; row++ ) {
            for ( size_t col = 0 ; col < columns ; col++ ) {
                if ( matrix(row, col) == infinity ) {
                    matrix(row, col) = max;
                }
            }
        }
    }
    static void minimize_along_direction(Matrix<Data> &matrix, const bool over_columns) {
        const size_t outer_size = over_columns ? matrix.columns() : matrix.rows(),
                     inner_size = over_columns ? matrix.rows() : matrix.columns();
        for ( size_t i = 0 ; i < outer_size ; i++ ) {
            double min = over_columns ? matrix(0, i) : matrix(i, 0);
            for ( size_t j = 1 ; j < inner_size && min > 0 ; j++ ) {
                min = XYZMIN(min, over_columns ? matrix(j, i) : matrix(i, j));
            }
            if ( min > 0 ) {
                for ( size_t j = 0 ; j < inner_size ; j++ ) {
                    if ( over_columns ) matrix(j, i) -= min;
                    else matrix(i, j) -= min;
                }
            }
        }
    }
private:
    inline bool find_uncovered_in_matrix(const double item, size_t &row, size_t &col) const {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        for ( row = 0 ; row < rows ; row++ ) {
            if ( !row_mask[row] ) {
                for ( col = 0 ; col < columns ; col++ ) {
                    if ( !col_mask[col] ) {
                        if ( matrix(row,col) == item ) return true;
                    }
                }
            }
        }
        return false;
    }
    bool pair_in_list(const std::pair<size_t,size_t> &needle, const std::list<std::pair<size_t,size_t> > &haystack) {
        for ( auto i = haystack.begin() ; i != haystack.end() ; i++ ) {
            if ( needle == *i ) return true;
        }
        return false;
    }
    int step1() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        for ( size_t row = 0 ; row < rows ; row++ ) {
            for ( size_t col = 0 ; col < columns ; col++ ) {
                if ( 0 == matrix(row, col) ) {
                    for ( size_t nrow = 0 ; nrow < row ; nrow++ )
                        if ( STAR == mask_matrix(nrow,col) )
                            goto next_column;
                    mask_matrix(row,col) = STAR;
                    goto next_row;
                }
                next_column:;
            }
            next_row:;
        }
        return 2;
    }
    int step2() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        size_t covercount = 0;
        for ( size_t row = 0 ; row < rows ; row++ )
            for ( size_t col = 0 ; col < columns ; col++ )
                if ( STAR == mask_matrix(row, col) ) {
                    col_mask[col] = true;
                    covercount++;
                }
        if ( covercount >= matrix.minsize() ) return 0;
        return 3;
    }
    int step3() {
        if ( find_uncovered_in_matrix(0, saverow, savecol) ) {
            mask_matrix(saverow,savecol) = PRIME;
        } else {
            return 5;
        }
        for ( size_t ncol = 0 ; ncol < matrix.columns() ; ncol++ ) {
            if ( mask_matrix(saverow,ncol) == STAR ) {
                row_mask[saverow] = true;
                col_mask[ncol] = false;
                return 3;
            }
        }
        return 4;
    }
    int step4() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        std::list<std::pair<size_t,size_t> > seq;
        std::pair<size_t,size_t> z0(saverow, savecol);
        seq.insert(seq.end(), z0);
        std::pair<size_t,size_t> z1(-1, -1), z2n(-1, -1);
        size_t row, col = savecol;
        bool madepair;
        do {
            madepair = false;
            for ( row = 0 ; row < rows ; row++ ) {
                if ( mask_matrix(row,col) == STAR ) {
                    z1.first = row;
                    z1.second = col;
                    if ( !pair_in_list(z1, seq) ) {
                        madepair = true;
                        seq.insert(seq.end(), z1);
                        break;
                    }
                }
            }
            if ( !madepair ) break;
            madepair = false;
            for ( col = 0 ; col < columns ; col++ ) {
                if ( mask_matrix(row, col) == PRIME ) {
                    z2n.first = row;
                    z2n.second = col;
                    if ( !pair_in_list(z2n, seq) ) {
                        madepair = true;
                        seq.insert(seq.end(), z2n);
                        break;
                    }
                }
            }
        } while ( madepair );
        for ( auto i = seq.begin() ; i != seq.end() ; i++ ) {
            if ( mask_matrix(i->first,i->second) == STAR )
                mask_matrix(i->first,i->second) = NORMAL;
            if ( mask_matrix(i->first,i->second) == PRIME )
                mask_matrix(i->first,i->second) = STAR;
        }
        for ( size_t row = 0 ; row < mask_matrix.rows() ; row++ )
            for ( size_t col = 0 ; col < mask_matrix.columns() ; col++ )
                if ( mask_matrix(row,col) == PRIME )
                    mask_matrix(row,col) = NORMAL;
        for ( size_t i = 0 ; i < rows ; i++ ) row_mask[i] = false;
        for ( size_t i = 0 ; i < columns ; i++ ) col_mask[i] = false;
        return 2;
    }
    int step5() {
        const size_t rows = matrix.rows(), columns = matrix.columns();
        double h = 100000;
        for ( size_t row = 0 ; row < rows ; row++ ) {
            if ( !row_mask[row] ) {
                for ( size_t col = 0 ; col < columns ; col++ ) {
                    if ( !col_mask[col] ) {
                        if ( h > matrix(row, col) && matrix(row, col) != 0 ) {
                            h = matrix(row, col);
                        }
                    }
                }
            }
        }
        for ( size_t row = 0 ; row < rows ; row++ ) {
            if ( row_mask[row] ) {
                for ( size_t col = 0 ; col < columns ; col++ ) {
                    matrix(row, col) += h;
                }
            }
        }
        for ( size_t col = 0 ; col < columns ; col++ ) {
            if ( !col_mask[col] ) {
                for ( size_t row = 0 ; row < rows ; row++ ) {
                    matrix(row, col) -= h;
                }
            }
        }
        return 3;
    }
    Matrix<int> mask_matrix;
    Matrix<Data> matrix;
    bool *row_mask;
    bool *col_mask;
    size_t saverow = 0, savecol = 0;
};

#endif /* !defined(_MUNKRES_H_) */
