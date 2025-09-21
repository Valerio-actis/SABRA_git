! --------------------------------------------------- |
! initialize seed for random sampling                 |

SUBROUTINE seedInit(iseed)
    
    INTEGER, INTENT(IN) :: iseed

    INTEGER              :: nseed
    INTEGER, ALLOCATABLE :: temp(:)
    INTEGER              :: j, k

    CALL RANDOM_SEED(size=k)
    nseed = MOD(iseed, ABS(HUGE(0)-iseed)-1)
    ALLOCATE (temp(k), source = nseed*[(j, j=0, k-1)])
    CALL RANDOM_SEED(put=temp)
    
    DEALLOCATE(temp)
    
END SUBROUTINE seedInit

! --------------------------------------------------- |
! sampling uniform distribution over [-1,1)           |

SUBROUTINE randuni(samp)

    DOUBLE PRECISION, INTENT(OUT) :: samp

    CALL RANDOM_NUMBER(samp)
    samp = (samp-0.5)*2

END SUBROUTINE randuni

